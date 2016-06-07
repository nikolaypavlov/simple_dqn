from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Gaussian
from neon.optimizers import RMSProp, Adam, Adadelta
from neon.layers import Affine, Conv, GeneralizedCost
from neon.transforms import Rectlin
from neon.models import Model
from neon.transforms import SumSquared
from neon.util.persist import save_obj
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class DeepQNetwork:
    def __init__(self, num_actions, args):
        # remember parameters
        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.history_length = args.history_length
        self.screen_dim = (args.screen_height, args.screen_width)
        self.clip_error = args.clip_error
        self.batch_norm = args.batch_norm
        self.double_dqn = args.double_dqn

        # create Neon backend
        self.be = gen_backend(backend=args.backend,
                                 batch_size=args.batch_size,
                                 rng_seed=args.random_seed,
                                 device_id=args.device_id,
                                 datatype=np.dtype(args.datatype).type,
                                 stochastic_round=args.stochastic_round)

        # prepare tensors once and reuse them
        # self.input_shape = (self.history_length,) + self.screen_dim + (self.batch_size,)
        self.input_shape = (self.history_length * max(self.screen_dim), ) + (self.batch_size,)
        self.input = self.be.empty(self.input_shape)
        self.input.lshape = self.input_shape # HACK: needed for convolutional networks
        self.targets = self.be.empty((self.num_actions, self.batch_size))

        # create model
        layers = self._createLayers(num_actions)
        self.model = Model(layers=layers)
        self.cost = GeneralizedCost(costfunc=SumSquared())
        # Bug fix
        for l in self.model.layers.layers:
            l.parallelism = 'Disabled'
        self.model.initialize(self.input_shape[:-1], self.cost)
        if args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(learning_rate=args.learning_rate,
                                     decay_rate=args.decay_rate,
                                     stochastic_round=args.stochastic_round)
        elif args.optimizer == 'adam':
            self.optimizer = Adam(learning_rate=args.learning_rate,
                                  stochastic_round=args.stochastic_round)
        elif args.optimizer == 'adadelta':
            self.optimizer = Adadelta(decay=args.decay_rate,
                                      stochastic_round=args.stochastic_round)
        else:
            assert False, "Unknown optimizer"

        # create target model
        self.target_steps = args.target_steps
        self.train_iterations = 0
        if self.target_steps:
            self.target_model = Model(layers=self._createLayers(num_actions))
            # Bug fix
            for l in self.target_model.layers.layers:
                l.parallelism = 'Disabled'
            self.target_model.initialize(self.input_shape[:-1])
            self.save_weights_prefix = args.save_weights_prefix
        else:
            self.target_model = self.model

        self.callback = None

    def _createLayers(self, num_actions):
        # create network
        init_norm = Gaussian(loc=0.0, scale=0.01)
        layers = []
        layers.append(Affine(nout=512, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
        # The output layer is a fully-connected linear layer with a single output for each valid action.
        layers.append(Affine(nout=num_actions, init=init_norm))
        return layers

    def _setInput(self, states):
        # change order of axes to match what Neon expects
        states = states.reshape(self.batch_size, -1)
        states = np.transpose(states, axes=(1, 0))
        # copy() shouldn't be necessary here, but Neon doesn't work otherwise
        self.input.set(states.copy())
        # normalize network input between 0 and 1
        # self.be.divide(self.input, 255, self.input)

    def train(self, minibatch, epoch):
        # expand components of minibatch
        prestates, actions, rewards, poststates, terminals = minibatch
        assert len(prestates.shape) == 4
        assert len(poststates.shape) == 4
        assert len(actions.shape) == 1
        assert len(rewards.shape) == 1
        assert len(terminals.shape) == 1
        assert prestates.shape == poststates.shape
        assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

        if self.target_steps and self.train_iterations % self.target_steps == 0:
            pdict = self.model.get_description(get_weights=True)
            self.target_model.deserialize(pdict, load_states=False)

        self._setInput(poststates)

        if self.double_dqn:
            # feed-forward pass for poststates to get argmax estimate
            postq_est = self.model.fprop(self.input, inference=True)
            max_idx = self.be.argmax(postq_est, axis=0).asnumpyarray().astype(np.intp)
            assert max_idx.shape == (1, self.batch_size)

            # feed-forward pass for poststates to get max Q-value by argmax estimate
            postq = self.target_model.fprop(self.input, inference=True).asnumpyarray()
            maxpostq = postq[max_idx[0], range(postq.shape[1])].reshape((1, -1))
        else:
            # feed-forward pass for poststates to get Q-values
            postq = self.target_model.fprop(self.input, inference=True)

            # calculate max Q-value for each poststate
            maxpostq = self.be.max(postq, axis=0).asnumpyarray()

        assert postq.shape == (self.num_actions, self.batch_size)
        assert maxpostq.shape == (1, self.batch_size)

        # feed-forward pass for prestates
        self._setInput(prestates)
        preq = self.model.fprop(self.input, inference=False)
        assert preq.shape == (self.num_actions, self.batch_size)

        # make copy of prestate Q-values as targets
        targets = np.copy(preq.asnumpyarray())

        # update Q-value targets for actions taken
        for i, action in enumerate(actions):
            if terminals[i]:
                targets[action, i] = float(rewards[i])
            else:
                targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[0,i]

        # copy targets to GPU memory
        self.targets.set(targets)

        # calculate errors
        deltas = self.cost.get_errors(preq, self.targets)
        assert deltas.shape == (self.num_actions, self.batch_size)
        # assert np.count_nonzero(deltas.asnumpyarray()) == 32

        # calculate cost, just in case
        cost = self.cost.get_cost(preq, self.targets)
        assert cost.shape == (1,1)

        # clip errors
        if self.clip_error:
            self.be.clip(deltas, -self.clip_error, self.clip_error, out=deltas)

        # perform back-propagation of gradients
        self.model.bprop(deltas)

        # perform optimization
        self.optimizer.optimize(self.model.layers_to_optimize, epoch)

        # increase number of weight updates (needed for target clone interval)
        self.train_iterations += 1

        # calculate statistics
        if self.callback:
            self.callback.on_train(cost.asnumpyarray()[0,0])

    def predict(self, states):
        # minibatch is full size, because Neon doesn't let change the minibatch size
        assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)

        # calculate Q-values for the states
        self._setInput(states)
        qvalues = self.model.fprop(self.input, inference=True)
        assert qvalues.shape == (self.num_actions, self.batch_size)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Q-values: " + str(qvalues.asnumpyarray()[:,0]))

        # transpose the result, so that batch size is first dimension
        return qvalues.T.asnumpyarray()

    def load_weights(self, load_path):
        self.model.load_weights(load_path)

    def save_weights(self, save_path):
        save_obj(self.model.serialize(keep_states=True), save_path)

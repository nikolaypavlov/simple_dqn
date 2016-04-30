import logging
logger = logging.getLogger(__name__)

import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, batch_norm
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import squared_error

class DQN:
    def __init__(self, num_actions, args):
        # remember parameters
        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.history_length = args.history_length
        self.screen_dim = (args.screen_height, args.screen_width)
        self.input_shape = (self.batch_size,) + (self.history_length,) + self.screen_dim
        self.decay_rate = args.decay_rate
        self.weight_decay = args.weight_decay
        self.clip_error = args.clip_error

        if args.optimizer == 'adadelta':
            self.optimizer = lasagne.updates.adadelta
            self.learning_rate = 1.0
        elif args.optimizer == 'rmsprop':
            self.optimizer = lasagne.updates.rmsprop
            self.learning_rate = args.learning_rate
        else:
            assert False, "Unknown optimizer"

        # create Theano model
        self.model, self.model_train, self.model_predict = self._create_network()

        # create target model
        self.target_steps = args.target_steps
        self.train_iterations = 0
        if self.target_steps:
            self.target_model, self.predict_target = self._create_target_network()
            self.save_weights_prefix = args.save_weights_prefix
        else:
            self.target_model = self.model

        self.callback = None

    def _build_network(self):
        l_in = InputLayer(self.input_shape, name="input")
        l_1 = Conv2DLayer(l_in, num_filters=32, filter_size=(8, 8), stride=4, nonlinearity=lasagne.nonlinearities.rectify, name="conv1")
        l_2 = Conv2DLayer(l_1, num_filters=64, filter_size=(4, 4), stride=2, nonlinearity=lasagne.nonlinearities.rectify, name="conv2")
        l_3 = Conv2DLayer(l_2, num_filters=64, filter_size=(3, 3), stride=1, nonlinearity=lasagne.nonlinearities.rectify, name="conv3")
        l_4 = DenseLayer(l_3, num_units=512, nonlinearity=lasagne.nonlinearities.rectify, name="fc1")
        l_out = DenseLayer(l_4, num_units=self.num_actions, nonlinearity=lasagne.nonlinearities.identity, W=lasagne.init.Normal(), name="out")

        return l_out, l_in.input_var

    def _create_target_network(self):
        logger.info("Building network with fixed weights...")
        net, input_var = self._build_network()
        predict = theano.function([input_var], lasagne.layers.get_output(net, input_var / 255.0))

        return net, predict

    def _create_network(self):
        logger.info("Building network ...")
        net, input_var = self._build_network()
        target_values = T.matrix('target_output')
        actions = T.icol('actions')

        # Create masks
        # mask = theano.shared(np.zeros((self.batch_size, self.num_actions)).astype(np.int32))
        mask = T.zeros_like(target_values)
        mask = T.set_subtensor(mask[T.arange(self.batch_size), actions.reshape((-1,))], 1)

        # feed-forward path
        network_output = lasagne.layers.get_output(net, input_var / 255.0)

        # Add regularization penalty
        loss = squared_error(network_output * mask, target_values).mean()
        if self.weight_decay > 0.0:
            loss += regularize_network_params(net, l2) * self.weight_decay

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(net, trainable=True)

        # Compute updates for training
        if self.clip_error:
            grads = theano.gradient.grad(loss, all_params)
            grads = [lasagne.updates.norm_constraint(grad, self.clip_error, range(grad.ndim)) for grad in grads]
            updates = self.optimizer(grads, all_params, learning_rate=self.learning_rate, rho=self.decay_rate)
        else:
            updates = self.optimizer(loss, all_params, learning_rate=self.learning_rate, rho=self.decay_rate)

        # Theano functions for training and computing cost
        logger.info("Compiling functions ...")
        train = theano.function([input_var, target_values, actions], [loss, network_output, target_values, mask], updates=updates)
        predict = theano.function([input_var], network_output)

        return net, train, predict

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
            self._sync_target_network()

        # feed-forward pass for poststates to get Q-values
        postq = self.predict_target(poststates)
        assert postq.shape == (self.batch_size, self.num_actions)

        # calculate max Q-value for each poststate
        maxpostq = np.max(postq, axis=1)
        assert maxpostq.shape == (self.batch_size, )

        targets = np.zeros_like(postq)
        for i, action in enumerate(actions):
            if terminals[i]:
                targets[i, action] = float(rewards[i])
            else:
                targets[i, action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        # train network
        loss, net_output, targets, mask = self.model_train(prestates, targets, actions.reshape((actions.shape[0], -1)))
        assert mask.sum() == self.batch_size == mask[range(self.batch_size), actions].sum()
        assert net_output.shape == (self.batch_size, self.num_actions)
        assert (~np.isclose(targets, net_output * mask)).sum() <= self.batch_size

        # increase number of weight updates (needed for target clone interval)
        self.train_iterations += 1

        # calculate statistics
        if self.callback:
            self.callback.on_train(loss.mean())

    def predict(self, states):
        assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)

        # calculate Q-values for the states
        qvalues = self.model_predict(states.astype(theano.config.floatX))
        assert qvalues.shape == (self.batch_size, self.num_actions)
        logger.debug("Q-values: %s" % qvalues)

        # transpose the result, so that batch size is first dimension
        return qvalues

    def save_weights(self, filename):
        f = file(filename, 'wb')
        pickle.dump(self._get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_weights(self, filename):
        f = file(filename, 'rb')
        self._set_param_values(pickle.load(f))
        f.close()

    def _get_param_values(self):
        t_values = lasagne.layers.get_all_param_values(self.target_model)
        values = lasagne.layers.get_all_param_values(self.model)
        return t_values, values

    def _set_param_values(self, params):
        lasagne.layers.set_all_param_values(self.target_model, params[0])
        lasagne.layers.set_all_param_values(self.model, params[1])

    def _sync_target_network(self):
        logger.debug("Syncing main network to target network")
        net_params = lasagne.layers.get_all_param_values(self.model)
        lasagne.layers.set_all_param_values(self.target_model, net_params)

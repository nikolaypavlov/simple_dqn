import sys
import os
import logging
import cv2
import socket
import numpy as np
logger = logging.getLogger(__name__)

class Environment:
    def __init__(self):
        pass

    def numActions(self):
        # Returns number of actions
        raise NotImplementedError

    def restart(self):
        # Restarts environment
        raise NotImplementedError

    def act(self, action):
        # Performs action and returns reward
        raise NotImplementedError

    def getScreen(self):
        # Gets current game screen
        raise NotImplementedError

    def isTerminal(self):
        # Returns if game is done
        raise NotImplementedError

class ALEEnvironment(Environment):
    def __init__(self, rom_file, args):
        from ale_python_interface import ALEInterface
        self.ale = ALEInterface()
        if args.display_screen:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                self.ale.setBool('sound', False) # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                self.ale.setBool('sound', True)
            self.ale.setBool('display_screen', True)

        self.ale.setInt('frame_skip', args.frame_skip)
        self.ale.setFloat('repeat_action_probability', args.repeat_action_probability)
        self.ale.setBool('color_averaging', args.color_averaging)

        if args.random_seed:
            self.ale.setInt('random_seed', args.random_seed)

        if args.record_screen_path:
            if not os.path.exists(args.record_screen_path):
                logger.info("Creating folder %s" % args.record_screen_path)
                os.makedirs(args.record_screen_path)
            logger.info("Recording screens to %s", args.record_screen_path)
            self.ale.setString('record_screen_dir', args.record_screen_path)

        if args.record_sound_filename:
            logger.info("Recording sound to %s", args.record_sound_filename)
            self.ale.setBool('sound', True)
            self.ale.setString('record_sound_filename', args.record_sound_filename)

        self.ale.loadROM(rom_file)

        if args.minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
            logger.info("Using minimal action set with size %d" % len(self.actions))
        else:
            self.actions = self.ale.getLegalActionSet()
            logger.info("Using full action set with size %d" % len(self.actions))
        logger.debug("Actions: " + str(self.actions))

        # OpenCV expects width as first and height as second
        self.dims = (args.screen_width, args.screen_height)

    def numActions(self):
        return len(self.actions)

    def restart(self):
        self.ale.reset_game()

    def act(self, action):
        reward = self.ale.act(self.actions[action])
        return reward

    def getScreen(self):
        screen = self.ale.getScreenGrayscale()
        import cv2
        resized = cv2.resize(screen, self.dims)
        return resized

    def isTerminal(self):
        return self.ale.game_over()

class GymEnvironment(Environment):
    # For use with Open AI Gym Environment
    def __init__(self, env_id, args):
        import gym
        self.gym = gym.make(env_id)
        self.obs = None
        self.terminal = None
        # OpenCV expects width as first and height as second s
        self.dims = (args.screen_width, args.screen_height)

    def numActions(self):
        import gym
        assert isinstance(self.gym.action_space, gym.spaces.Discrete)
        return self.gym.action_space.n

    def restart(self):
        self.obs = self.gym.reset()
        self.terminal = False

    def act(self, action):
        self.obs, reward, self.terminal, _ = self.gym.step(action)
        return reward

    def getScreen(self):
        assert self.obs is not None
        return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), self.dims)

    def isTerminal(self):
        assert self.terminal is not None
        return self.terminal

class F9Environment(Environment):
    def __init__(self, args):
        self.socket = socket.socket()
        self.socket.connect((args.ip, args.port))
        self.actions = [[0,0,0,0], [1,0,1,0], [1,1,0,0], [1,1,1,0]]
        self.obs = None
        self.frame_skip = args.frame_skip

    def _getObservation(self):
        # getting data from server
        data = eval(self.socket.recv(1024))
        self.obs = None
        if data:
            agent = (item for item in data if item["type"] == "actor").next()
            platform = (item for item in data if item["type"] == "decoration").next()
            system = (item for item in data if item["type"] == "system").next()
            self.obs = (agent, platform, system)

        return self.obs

    def _getReward(self):
        assert self.obs is not None
        agent, platform, system = self.obs
        if system["flight_status"] == "landed":
            reward = 1.0
        elif self.isTerminal():
            reward = -1.0
        else:  # Remove this if you don't want to use handcrafted heuristic
            reward = 1.0 / (1 + (agent["px"] - platform["px"]) ** 2 + (agent["py"] - agent["py"]) ** 2) + agent["contact_time"]

        return reward

    def numActions(self):
        return len(self.actions)

    def restart(self):
        self.socket.send(str([0, 0, 0, 1]))
        self.obs = self._getObservation()

    def act(self, action):
        # act in the game environment
        for f in range(self.frame_skip + 1):
            self.socket.send(str(self.actions[action]))
            self.obs = self._getObservation()
        return self._getReward()

    def getScreen(self):
        assert self.obs is not None
        agent, platform, _ = self.obs
        features = np.array([agent['angle'],
                             (agent['py'] - platform['py']) ** 2,
                             (agent['px'] - platform['px']) ** 2,
                             np.sign(agent['wind']),
                             float(agent['fuel'] > 0)],
                             dtype=np.float32)
        return features.reshape((-1, 1))

    def isTerminal(self):
        # system["flight_status"] | "none", "landed", "destroyed"
        # "none" means that we don't know, whether we landed or destroyed
        agent, _, system = self.obs
        terminal = False
        if system["flight_status"] == "destroyed" or system["flight_status"] == "landed" or agent["py"] <= 0.0:
            terminal = True
        return terminal

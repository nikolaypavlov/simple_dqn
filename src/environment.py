import sys
import os
from ale_python_interface import ALEInterface
from PIL import Image
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Environment:
    def __init__(self, rom_file, args):
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
        img = Image.fromarray(screen.reshape(screen.shape[0:2]))
        resized = img.resize(self.dims, Image.BICUBIC)
        return np.array(resized, dtype=np.uint8)

    def isTerminal(self):
        return self.ale.game_over()

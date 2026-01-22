import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_num_channels(self):
    """
        Tests if Sound.get_num_channels returns the correct number
        of channels playing a specific sound.
        """
    try:
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound = mixer.Sound(file=filename)
        self.assertEqual(sound.get_num_channels(), 0)
        sound.play()
        self.assertEqual(sound.get_num_channels(), 1)
        sound.play()
        self.assertEqual(sound.get_num_channels(), 2)
        sound.stop()
        self.assertEqual(sound.get_num_channels(), 0)
    finally:
        pygame.mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            sound.get_num_channels()
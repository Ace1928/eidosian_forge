import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_all_sounds_stopped_with_fadeout(self):
    """
        Test that get_busy returns False when all sounds are stopped with
        fadeout.
        """
    sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound1.play()
    sound2.play()
    time.sleep(0.2)
    sound1.fadeout(100)
    sound2.fadeout(100)
    time.sleep(0.3)
    self.assertFalse(pygame.mixer.get_busy())
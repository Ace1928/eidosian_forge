import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_multiple_sounds_playing(self):
    """
        Test that get_busy returns True when multiple sounds are playing.
        """
    sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound1.play()
    sound2.play()
    time.sleep(0.2)
    self.assertTrue(pygame.mixer.get_busy())
    sound1.stop()
    sound2.stop()
import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound__from_sound_object(self):
    """Ensure Sound() creation with a Sound() object works."""
    filename = example_path(os.path.join('data', 'house_lo.wav'))
    sound_obj = mixer.Sound(file=filename)
    sound = mixer.Sound(sound_obj)
    self.assertIsInstance(sound, mixer.Sound)
import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_no_sound_playing(self):
    """
        Test that get_busy returns False when no sound is playing.
        """
    self.assertFalse(pygame.mixer.get_busy())
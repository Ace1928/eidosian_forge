import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_set_reserved(self):
    """Ensure pygame.mixer.set_reserved() reserves the given number of channels."""
    mixer.init()
    default_num_channels = mixer.get_num_channels()
    result = mixer.set_reserved(default_num_channels)
    self.assertEqual(result, default_num_channels)
    result = mixer.set_reserved(default_num_channels + 1)
    self.assertEqual(result, default_num_channels)
    result = mixer.set_reserved(0)
    self.assertEqual(result, 0)
    result = mixer.set_reserved(int(default_num_channels / 2))
    self.assertEqual(result, int(default_num_channels / 2))
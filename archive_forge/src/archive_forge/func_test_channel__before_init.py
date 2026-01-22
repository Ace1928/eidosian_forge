import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_channel__before_init(self):
    """Ensure exception for Channel() creation with non-init mixer."""
    mixer.quit()
    with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
        mixer.Channel(0)
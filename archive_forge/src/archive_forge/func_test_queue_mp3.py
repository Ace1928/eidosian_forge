import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue_mp3(self):
    """Ensures queue() accepts mp3 files.

        |tags:music|
        """
    filename = example_path(os.path.join('data', 'house_lo.mp3'))
    pygame.mixer.music.queue(filename)
import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue__multiple_calls(self):
    """Ensures queue() can be called multiple times."""
    ogg_file = example_path(os.path.join('data', 'house_lo.ogg'))
    wav_file = example_path(os.path.join('data', 'house_lo.wav'))
    pygame.mixer.music.queue(ogg_file)
    pygame.mixer.music.queue(wav_file)
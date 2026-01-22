import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_queue__arguments(self):
    """Ensures queue() can be called with proper arguments."""
    wav_file = example_path(os.path.join('data', 'house_lo.wav'))
    pygame.mixer.music.queue(wav_file, loops=2)
    pygame.mixer.music.queue(wav_file, namehint='')
    pygame.mixer.music.queue(wav_file, '')
    pygame.mixer.music.queue(wav_file, '', 2)
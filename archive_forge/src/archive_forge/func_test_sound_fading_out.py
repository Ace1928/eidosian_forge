import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound_fading_out(self):
    """Tests that get_busy() returns True when a sound is fading out"""
    sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    sound.play(fade_ms=1000)
    time.sleep(1.1)
    self.assertTrue(pygame.mixer.get_busy())
    sound.stop()
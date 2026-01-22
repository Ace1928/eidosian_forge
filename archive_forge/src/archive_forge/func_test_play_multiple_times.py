import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_play_multiple_times(self):
    """Test playing a sound multiple times."""
    frequency, format, channels = mixer.get_init()
    sound_length_in_ms = 100
    bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
    sound = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
    self.assertAlmostEqual(sound.get_length(), sound_length_in_ms / 1000.0, places=2)
    num_loops = 5
    channel = sound.play(loops=num_loops)
    self.assertIsInstance(channel, pygame.mixer.Channel)
    pygame.time.wait(sound_length_in_ms * num_loops - 100)
    self.assertTrue(channel.get_busy())
    pygame.time.wait(sound_length_in_ms + 200)
    self.assertFalse(channel.get_busy())
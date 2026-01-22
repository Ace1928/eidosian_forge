import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_play_indefinitely(self):
    """Test playing a sound indefinitely."""
    frequency, format, channels = mixer.get_init()
    sound_length_in_ms = 100
    bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
    sound = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
    channel = sound.play(loops=-1)
    self.assertIsInstance(channel, pygame.mixer.Channel)
    for _ in range(2):
        self.assertTrue(channel.get_busy())
        pygame.time.wait(sound_length_in_ms)
import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_length(self):
    """Tests if get_length returns a correct length."""
    try:
        for size in SIZES:
            pygame.mixer.quit()
            pygame.mixer.init(size=size)
            filename = example_path(os.path.join('data', 'punch.wav'))
            sound = mixer.Sound(file=filename)
            sound_bytes = sound.get_raw()
            mix_freq, mix_bits, mix_channels = pygame.mixer.get_init()
            mix_bytes = abs(mix_bits) / 8
            expected_length = float(len(sound_bytes)) / mix_freq / mix_bytes / mix_channels
            self.assertAlmostEqual(expected_length, sound.get_length())
    finally:
        pygame.mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            sound.get_length()
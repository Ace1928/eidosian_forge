import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_find_channel(self):
    mixer.init()
    filename = example_path(os.path.join('data', 'house_lo.wav'))
    sound = mixer.Sound(file=filename)
    num_channels = mixer.get_num_channels()
    if num_channels > 0:
        found_channel = mixer.find_channel()
        self.assertIsNotNone(found_channel)
        channels = []
        for channel_id in range(0, num_channels):
            channel = mixer.Channel(channel_id)
            channel.play(sound)
            channels.append(channel)
        found_channel = mixer.find_channel()
        self.assertIsNone(found_channel)
        found_channel = mixer.find_channel(True)
        self.assertIsNotNone(found_channel)
        found_channel = mixer.find_channel(force=True)
        self.assertIsNotNone(found_channel)
        for channel in channels:
            channel.stop()
        found_channel = mixer.find_channel()
        self.assertIsNotNone(found_channel)
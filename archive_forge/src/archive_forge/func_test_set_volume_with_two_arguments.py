import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
@unittest.expectedFailure
def test_set_volume_with_two_arguments(self):
    self.channel.play(self.sound)
    self.channel.set_volume(0.3, 0.7)
    self.assertEqual(self.channel.get_volume(), (0.3, 0.7))
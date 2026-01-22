import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_set_volume_with_one_argument(self):
    self.channel.play(self.sound)
    self.channel.set_volume(0.5)
    self.assertEqual(self.channel.get_volume(), 0.5)
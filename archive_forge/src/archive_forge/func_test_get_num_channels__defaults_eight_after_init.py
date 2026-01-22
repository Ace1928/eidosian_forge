import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_num_channels__defaults_eight_after_init(self):
    mixer.init()
    self.assertEqual(mixer.get_num_channels(), 8)
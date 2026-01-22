import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_init__zero_values(self):
    mixer.pre_init(44100, 8, 1, allowedchanges=0)
    mixer.init(0, 0, 0)
    self.assertEqual(mixer.get_init(), (44100, 8, 1))
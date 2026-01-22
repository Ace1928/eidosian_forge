import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_init__returns_None_if_mixer_not_initialized(self):
    self.assertIsNone(mixer.get_init())
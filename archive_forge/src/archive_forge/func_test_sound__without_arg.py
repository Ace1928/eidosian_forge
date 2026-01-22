import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound__without_arg(self):
    """Ensure exception raised for Sound() creation with no argument."""
    with self.assertRaises(TypeError):
        mixer.Sound()
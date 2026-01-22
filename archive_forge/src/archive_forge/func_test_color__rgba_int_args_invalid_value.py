import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__rgba_int_args_invalid_value(self):
    """Ensures invalid values are detected when creating Color objects."""
    self.assertRaises(ValueError, pygame.Color, 257, 10, 105, 44)
    self.assertRaises(ValueError, pygame.Color, 10, 257, 105, 44)
    self.assertRaises(ValueError, pygame.Color, 10, 105, 257, 44)
    self.assertRaises(ValueError, pygame.Color, 10, 105, 44, 257)
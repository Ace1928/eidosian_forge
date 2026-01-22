import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__rgba_int_args_invalid_value_without_alpha(self):
    """Ensures invalid values are detected when creating Color objects
        without providing an alpha.
        """
    self.assertRaises(ValueError, pygame.Color, 256, 10, 105)
    self.assertRaises(ValueError, pygame.Color, 10, 256, 105)
    self.assertRaises(ValueError, pygame.Color, 10, 105, 256)
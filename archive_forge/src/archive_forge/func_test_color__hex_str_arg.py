import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__hex_str_arg(self):
    """Ensures Color objects can be created using hex strings."""
    color = pygame.Color('0x1a2B3c4D')
    self.assertEqual(color.r, 26)
    self.assertEqual(color.g, 43)
    self.assertEqual(color.b, 60)
    self.assertEqual(color.a, 77)
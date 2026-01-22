import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__rgba_int_args(self):
    """Ensures Color objects can be created using ints."""
    color = pygame.Color(10, 20, 30, 40)
    self.assertEqual(color.r, 10)
    self.assertEqual(color.g, 20)
    self.assertEqual(color.b, 30)
    self.assertEqual(color.a, 40)
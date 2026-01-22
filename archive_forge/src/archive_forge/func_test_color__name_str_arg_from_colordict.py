import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__name_str_arg_from_colordict(self):
    """Ensures Color objects can be created using str names
        from the THECOLORS dict."""
    for name, values in THECOLORS.items():
        color = pygame.Color(name)
        self.assertEqual(color.r, values[0])
        self.assertEqual(color.g, values[1])
        self.assertEqual(color.b, values[2])
        self.assertEqual(color.a, values[3])
import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__name_str_arg(self):
    """Ensures Color objects can be created using str names."""
    for name in ('aquamarine3', 'AQUAMARINE3', 'AqUAmArIne3'):
        color = pygame.Color(name)
        self.assertEqual(color.r, 102)
        self.assertEqual(color.g, 205)
        self.assertEqual(color.b, 170)
        self.assertEqual(color.a, 255)
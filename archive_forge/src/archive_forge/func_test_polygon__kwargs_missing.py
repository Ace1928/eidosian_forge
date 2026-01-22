import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def test_polygon__kwargs_missing(self):
    """Ensures draw polygon detects any missing required kwargs."""
    kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'points': ((2, 1), (2, 2), (2, 3)), 'width': 1}
    for name in ('points', 'color', 'surface'):
        invalid_kwargs = dict(kwargs)
        invalid_kwargs.pop(name)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(**invalid_kwargs)
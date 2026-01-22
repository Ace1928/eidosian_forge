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
def test_circle__kwargs_missing(self):
    """Ensures draw circle detects any missing required kwargs."""
    kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'center': (1, 0), 'radius': 2, 'width': 1, 'draw_top_right': False, 'draw_top_left': False, 'draw_bottom_left': False, 'draw_bottom_right': True}
    for name in ('radius', 'center', 'color', 'surface'):
        invalid_kwargs = dict(kwargs)
        invalid_kwargs.pop(name)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(**invalid_kwargs)
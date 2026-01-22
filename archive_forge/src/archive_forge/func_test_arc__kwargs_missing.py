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
def test_arc__kwargs_missing(self):
    """Ensures draw arc detects any missing required kwargs."""
    kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'rect': pygame.Rect((1, 0), (2, 2)), 'start_angle': 0.1, 'stop_angle': 2, 'width': 1}
    for name in ('stop_angle', 'start_angle', 'rect', 'color', 'surface'):
        invalid_kwargs = dict(kwargs)
        invalid_kwargs.pop(name)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(**invalid_kwargs)
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
def test_lines__invalid_points_formats(self):
    """Ensures draw lines handles invalid points formats correctly."""
    kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None, 'width': 1}
    points_fmts = (((1, 1), (2,)), ((1, 1), (2, 2, 2)), ((1, 1), (2, '2')), ((1, 1), {2, 3}), ((1, 1), dict(((2, 2), (3, 3)))), {(1, 1), (1, 2)}, dict(((1, 1), (4, 4))))
    for points in points_fmts:
        kwargs['points'] = points
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(**kwargs)
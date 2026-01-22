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
def test_aalines__kwarg_invalid_name(self):
    """Ensures draw aalines detects invalid kwarg names."""
    surface = pygame.Surface((2, 3))
    color = pygame.Color('cyan')
    closed = 1
    points = ((1, 2), (2, 1))
    kwargs_list = [{'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}, {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(**kwargs)
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
def test_rect__kwarg_invalid_name(self):
    """Ensures draw rect detects invalid kwarg names."""
    surface = pygame.Surface((2, 1))
    color = pygame.Color('green')
    rect = pygame.Rect((0, 0), (3, 3))
    kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'invalid': 1}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(**kwargs)
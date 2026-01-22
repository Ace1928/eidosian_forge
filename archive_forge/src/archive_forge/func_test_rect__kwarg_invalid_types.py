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
def test_rect__kwarg_invalid_types(self):
    """Ensures draw rect detects invalid kwarg types."""
    surface = pygame.Surface((2, 3))
    color = pygame.Color('red')
    rect = pygame.Rect((0, 0), (1, 1))
    kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': 2.3, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': (1, 1, 2), 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1.1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10.5, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5.5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 'a', 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 'c', 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 'd'}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(**kwargs)
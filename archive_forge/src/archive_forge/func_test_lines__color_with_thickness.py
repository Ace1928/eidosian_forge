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
def test_lines__color_with_thickness(self):
    """Ensures thick lines are drawn using the correct color."""
    x_left = y_top = 5
    for surface in self._create_surfaces():
        x_right = surface.get_width() - 5
        y_bottom = surface.get_height() - 5
        endpoints = ((x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom))
        for expected_color in self.COLORS:
            self.draw_lines(surface, expected_color, True, endpoints, 3)
            for t in (-1, 0, 1):
                for x in range(x_left, x_right + 1):
                    for y in (y_top, y_bottom):
                        pos = (x, y + t)
                        self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
                for y in range(y_top, y_bottom + 1):
                    for x in (x_left, x_right):
                        pos = (x + t, y)
                        self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
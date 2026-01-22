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
def test_polygon__valid_width_values(self):
    """Ensures draw polygon accepts different width values."""
    surface_color = pygame.Color('white')
    surface = pygame.Surface((3, 4))
    color = (10, 20, 30, 255)
    kwargs = {'surface': surface, 'color': color, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': None}
    pos = kwargs['points'][0]
    for width in (-100, -10, -1, 0, 1, 10, 100):
        surface.fill(surface_color)
        kwargs['width'] = width
        expected_color = color if width >= 0 else surface_color
        bounds_rect = self.draw_polygon(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)
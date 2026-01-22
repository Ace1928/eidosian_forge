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
def test_polygon__valid_color_formats(self):
    """Ensures draw polygon accepts different color formats."""
    green_color = pygame.Color('green')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((3, 4))
    kwargs = {'surface': surface, 'color': None, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': 0}
    pos = kwargs['points'][0]
    greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
    for color in greens:
        surface.fill(surface_color)
        kwargs['color'] = color
        if isinstance(color, int):
            expected_color = surface.unmap_rgb(color)
        else:
            expected_color = green_color
        bounds_rect = self.draw_polygon(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)
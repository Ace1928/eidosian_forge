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
def test_circle__valid_color_formats(self):
    """Ensures draw circle accepts different color formats."""
    center = (2, 2)
    radius = 1
    pos = (center[0] - radius, center[1])
    green_color = pygame.Color('green')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((3, 4))
    kwargs = {'surface': surface, 'color': None, 'center': center, 'radius': radius, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
    greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
    for color in greens:
        surface.fill(surface_color)
        kwargs['color'] = color
        if isinstance(color, int):
            expected_color = surface.unmap_rgb(color)
        else:
            expected_color = green_color
        bounds_rect = self.draw_circle(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)
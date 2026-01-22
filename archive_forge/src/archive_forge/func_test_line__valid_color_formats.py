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
def test_line__valid_color_formats(self):
    """Ensures draw line accepts different color formats."""
    green_color = pygame.Color('green')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((3, 4))
    pos = (1, 1)
    kwargs = {'surface': surface, 'color': None, 'start_pos': pos, 'end_pos': (2, 1), 'width': 3}
    greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
    for color in greens:
        surface.fill(surface_color)
        kwargs['color'] = color
        if isinstance(color, int):
            expected_color = surface.unmap_rgb(color)
        else:
            expected_color = green_color
        bounds_rect = self.draw_line(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)
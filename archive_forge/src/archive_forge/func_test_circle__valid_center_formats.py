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
def test_circle__valid_center_formats(self):
    """Ensures draw circle accepts different center formats."""
    expected_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((4, 4))
    kwargs = {'surface': surface, 'color': expected_color, 'center': None, 'radius': 1, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
    x, y = (2, 2)
    for center in ((x, y), (x + 0.1, y), (x, y + 0.1), (x + 0.1, y + 0.1)):
        for seq_type in (tuple, list, Vector2):
            surface.fill(surface_color)
            kwargs['center'] = seq_type(center)
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at((x, y)), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)
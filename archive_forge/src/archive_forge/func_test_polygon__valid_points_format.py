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
def test_polygon__valid_points_format(self):
    """Ensures draw polygon accepts different points formats."""
    expected_color = (10, 20, 30, 255)
    surface_color = pygame.Color('white')
    surface = pygame.Surface((3, 4))
    kwargs = {'surface': surface, 'color': expected_color, 'points': None, 'width': 0}
    point_types = ((tuple, tuple, tuple, tuple), (list, list, list, list), (Vector2, Vector2, Vector2, Vector2), (list, Vector2, tuple, Vector2))
    point_values = (((1, 1), (2, 1), (2, 2), (1, 2)), ((1, 1), (2.2, 1), (2.1, 2.2), (1, 2.1)))
    seq_types = (tuple, list)
    for point_type in point_types:
        for values in point_values:
            check_pos = values[0]
            points = [point_type[i](pt) for i, pt in enumerate(values)]
            for seq_type in seq_types:
                surface.fill(surface_color)
                kwargs['points'] = seq_type(points)
                bounds_rect = self.draw_polygon(**kwargs)
                self.assertEqual(surface.get_at(check_pos), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)
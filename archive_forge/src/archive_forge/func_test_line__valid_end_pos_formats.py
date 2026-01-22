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
def test_line__valid_end_pos_formats(self):
    """Ensures draw line accepts different end_pos formats."""
    expected_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((4, 4))
    kwargs = {'surface': surface, 'color': expected_color, 'start_pos': (2, 1), 'end_pos': None, 'width': 2}
    x, y = (2, 2)
    for end_pos in ((x, y), (x + 0.2, y), (x, y + 0.2), (x + 0.2, y + 0.2)):
        for seq_type in (tuple, list, Vector2):
            surface.fill(surface_color)
            kwargs['end_pos'] = seq_type(end_pos)
            bounds_rect = self.draw_line(**kwargs)
            self.assertEqual(surface.get_at((x, y)), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)
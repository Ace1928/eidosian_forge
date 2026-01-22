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
def test_line__bounding_rect(self):
    """Ensures draw line returns the correct bounding rect.

        Tests lines with endpoints on and off the surface and a range of
        width/thickness values.
        """
    if isinstance(self, PythonDrawTestCase):
        self.skipTest('bounding rects not supported in draw_py.draw_line')
    line_color = pygame.Color('red')
    surf_color = pygame.Color('black')
    width = height = 30
    helper_rect = pygame.Rect((0, 0), (width, height))
    for size in ((width + 5, height + 5), (width - 5, height - 5)):
        surface = pygame.Surface(size, 0, 32)
        surf_rect = surface.get_rect()
        for pos in rect_corners_mids_and_center(surf_rect):
            helper_rect.center = pos
            for thickness in range(-1, 5):
                for start, end in self._rect_lines(helper_rect):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_line(surface, line_color, start, end, thickness)
                    if 0 < thickness:
                        expected_rect = create_bounding_rect(surface, surf_color, start)
                    else:
                        expected_rect = pygame.Rect(start, (0, 0))
                    self.assertEqual(bounding_rect, expected_rect, 'start={}, end={}, size={}, thickness={}'.format(start, end, size, thickness))
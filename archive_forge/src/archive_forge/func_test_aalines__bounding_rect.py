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
def test_aalines__bounding_rect(self):
    """Ensures draw aalines returns the correct bounding rect.

        Tests lines with endpoints on and off the surface and blending
        enabled and disabled.
        """
    line_color = pygame.Color('red')
    surf_color = pygame.Color('blue')
    width = height = 30
    pos_rect = pygame.Rect((0, 0), (width, height))
    for size in ((width + 5, height + 5), (width - 5, height - 5)):
        surface = pygame.Surface(size, 0, 32)
        surf_rect = surface.get_rect()
        for pos in rect_corners_mids_and_center(surf_rect):
            pos_rect.center = pos
            pts = (pos_rect.midleft, pos_rect.midtop, pos_rect.midright)
            pos = pts[0]
            for closed in (True, False):
                surface.fill(surf_color)
                bounding_rect = self.draw_aalines(surface, line_color, closed, pts)
                expected_rect = create_bounding_rect(surface, surf_color, pos)
                self.assertEqual(bounding_rect, expected_rect)
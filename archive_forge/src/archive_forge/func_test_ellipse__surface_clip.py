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
def test_ellipse__surface_clip(self):
    """Ensures draw ellipse respects a surface's clip area.

        Tests drawing the ellipse filled and unfilled.
        """
    surfw = surfh = 30
    ellipse_color = pygame.Color('red')
    surface_color = pygame.Color('green')
    surface = pygame.Surface((surfw, surfh))
    surface.fill(surface_color)
    clip_rect = pygame.Rect((0, 0), (11, 11))
    clip_rect.center = surface.get_rect().center
    pos_rect = clip_rect.copy()
    for width in (0, 1):
        for center in rect_corners_mids_and_center(clip_rect):
            pos_rect.center = center
            surface.set_clip(None)
            surface.fill(surface_color)
            self.draw_ellipse(surface, ellipse_color, pos_rect, width)
            expected_pts = get_color_points(surface, ellipse_color, clip_rect)
            surface.fill(surface_color)
            surface.set_clip(clip_rect)
            self.draw_ellipse(surface, ellipse_color, pos_rect, width)
            surface.lock()
            for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                if pt in expected_pts:
                    expected_color = ellipse_color
                else:
                    expected_color = surface_color
                self.assertEqual(surface.get_at(pt), expected_color, pt)
            surface.unlock()
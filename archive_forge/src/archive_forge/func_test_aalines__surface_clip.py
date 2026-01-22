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
def test_aalines__surface_clip(self):
    """Ensures draw aalines respects a surface's clip area."""
    surfw = surfh = 30
    aaline_color = pygame.Color('red')
    surface_color = pygame.Color('green')
    surface = pygame.Surface((surfw, surfh))
    surface.fill(surface_color)
    clip_rect = pygame.Rect((0, 0), (11, 11))
    clip_rect.center = surface.get_rect().center
    pos_rect = clip_rect.copy()
    for center in rect_corners_mids_and_center(clip_rect):
        pos_rect.center = center
        pts = (pos_rect.midtop, pos_rect.center, pos_rect.midbottom)
        for closed in (True, False):
            surface.set_clip(None)
            surface.fill(surface_color)
            self.draw_aalines(surface, aaline_color, closed, pts)
            expected_pts = get_color_points(surface, surface_color, clip_rect, False)
            surface.fill(surface_color)
            surface.set_clip(clip_rect)
            self.draw_aalines(surface, aaline_color, closed, pts)
            surface.lock()
            for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                if pt in expected_pts:
                    self.assertNotEqual(surface.get_at(pt), surface_color, pt)
                else:
                    self.assertEqual(surface.get_at(pt), surface_color, pt)
            surface.unlock()
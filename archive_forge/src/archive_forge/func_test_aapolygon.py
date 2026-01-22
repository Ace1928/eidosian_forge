import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_aapolygon(self):
    """aapolygon(surface, points, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    points = [(10, 80), (10, 15), (92, 25), (92, 80)]
    fg_test_points = points
    bg_test_points = [(points[0][0] - 1, points[0][1]), (points[0][0], points[0][1] + 1), (points[0][0] - 1, points[0][1] + 1), (points[0][0] + 1, points[0][1] - 1), (points[3][0] + 1, points[3][1]), (points[3][0], points[3][1] + 1), (points[3][0] + 1, points[3][1] + 1), (points[3][0] - 1, points[3][1] - 1), (points[2][0] + 1, points[2][1]), (points[2][0] - 1, points[2][1] + 1), (points[1][0] - 1, points[1][1]), (points[1][0], points[1][1] - 1), (points[1][0] - 1, points[1][1] - 1)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.aapolygon(surf, points, fg)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_not_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)
import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_aaellipse(self):
    """aaellipse(surface, x, y, rx, ry, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    x = 45
    y = 40
    rx = 30
    ry = 35
    fg_test_points = [(x, y - ry), (x, y + ry), (x - rx, y), (x + rx, y)]
    bg_test_points = [(x, y), (x, y - ry + 1), (x, y - ry - 1), (x, y + ry + 1), (x, y + ry - 1), (x - rx - 1, y), (x - rx + 1, y), (x + rx + 1, y), (x + rx - 1, y)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.aaellipse(surf, x, y, rx, ry, fg)
        for posn in fg_test_points:
            self.check_not_at(surf, posn, bg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)
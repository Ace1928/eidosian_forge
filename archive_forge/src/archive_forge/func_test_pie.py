import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_pie(self):
    """pie(surface, x, y, r, start, end, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    x = 45
    y = 40
    r = 30
    start = 0
    end = 90
    fg_test_points = [(x, y), (x + 1, y), (x, y + 1), (x + r, y)]
    bg_test_points = [(x - 1, y), (x, y - 1), (x - 1, y - 1), (x + 1, y + 1), (x + r + 1, y), (x + r, y - 1), (x, y + r + 1)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.pie(surf, x, y, r, start, end, fg)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)
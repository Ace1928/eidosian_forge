import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_pixel(self):
    """pixel(surface, x, y, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.pixel(surf, 2, 2, fg)
        for x in range(1, 4):
            for y in range(1, 4):
                if x == 2 and y == 2:
                    self.check_at(surf, (x, y), fg_adjusted)
                else:
                    self.check_at(surf, (x, y), bg_adjusted)
import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_array3d(self):
    sources = [self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
    palette = self.test_palette
    for surf in sources:
        arr = pygame.surfarray.array3d(surf)

        def same_color(ac, sc):
            return ac[0] == sc[0] and ac[1] == sc[1] and (ac[2] == sc[2])
        for posn, i in self.test_points:
            self.assertTrue(same_color(arr[posn], surf.get_at(posn)), '%s != %s: flags: %i, bpp: %i, posn: %s' % (tuple(arr[posn]), surf.get_at(posn), surf.get_flags(), surf.get_bitsize(), posn))
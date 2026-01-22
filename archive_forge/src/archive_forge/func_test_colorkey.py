import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
@unittest.skipIf('ppc64le' in platform.uname(), 'known ppc64le issue')
def test_colorkey(self):
    pygame.display.set_mode((100, 50))
    bitsizes = [8, 16, 24, 32]
    for bitsize in bitsizes:
        surf = self._make_surface(bitsize)
        surf.set_colorkey(self._test_palette[1])
        surf.blit(surf, (3, 0))
        p = []
        for c in self._test_palette:
            c = surf.unmap_rgb(surf.map_rgb(c))
            p.append(c)
        p[1] = (p[1][0], p[1][1], p[1][2], 0)
        tmp = self._make_surface(32, srcalpha=True, palette=p)
        tmp.blit(tmp, (3, 0))
        tmp.set_alpha(None)
        comp = self._make_surface(bitsize)
        comp.blit(tmp, (0, 0))
        self._assert_same(surf, comp)
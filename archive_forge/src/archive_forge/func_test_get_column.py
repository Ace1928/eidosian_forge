import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_get_column(self):
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((6, 8), 0, bpp)
        sf.fill((0, 0, 255))
        val = sf.map_rgb((0, 0, 255))
        ar = pygame.PixelArray(sf)
        ar2 = ar.__getitem__(1)
        self.assertEqual(len(ar2), 8)
        self.assertEqual(ar2.__getitem__(0), val)
        self.assertEqual(ar2.__getitem__(1), val)
        self.assertEqual(ar2.__getitem__(2), val)
        ar2 = ar.__getitem__(-1)
        self.assertEqual(len(ar2), 8)
        self.assertEqual(ar2.__getitem__(0), val)
        self.assertEqual(ar2.__getitem__(1), val)
        self.assertEqual(ar2.__getitem__(2), val)
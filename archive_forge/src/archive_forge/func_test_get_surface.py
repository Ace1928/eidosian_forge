import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_get_surface(self):
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((10, 20), 0, bpp)
        sf.fill((0, 0, 0))
        ar = pygame.PixelArray(sf)
        self.assertTrue(ar.surface is sf)
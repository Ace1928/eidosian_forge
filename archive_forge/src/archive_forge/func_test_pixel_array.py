import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_pixel_array(self):
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((10, 20), 0, bpp)
        sf.fill((0, 0, 0))
        ar = pygame.PixelArray(sf)
        self.assertEqual(ar._pixels_address, sf._pixels_address)
        if sf.mustlock():
            self.assertTrue(sf.get_locked())
        self.assertEqual(len(ar), 10)
        del ar
        if sf.mustlock():
            self.assertFalse(sf.get_locked())
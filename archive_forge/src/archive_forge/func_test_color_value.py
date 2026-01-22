import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_color_value(self):
    sf = pygame.Surface((5, 5), 0, 32)
    ar = pygame.PixelArray(sf)
    index = slice(None, None, 1)
    ar.__setitem__(index, (1, 2, 3))
    self.assertEqual(ar[0, 0], sf.map_rgb((1, 2, 3)))
    ar.__setitem__(index, pygame.Color(10, 11, 12))
    self.assertEqual(ar[0, 0], sf.map_rgb((10, 11, 12)))
    self.assertRaises(ValueError, ar.__setitem__, index, (1, 2, 3, 4, 5))
    self.assertRaises(ValueError, ar.__setitem__, (index, index), (1, 2, 3, 4, 5))
    self.assertRaises(ValueError, ar.__setitem__, index, [1, 2, 3])
    self.assertRaises(ValueError, ar.__setitem__, (index, index), [1, 2, 3])
    sf = pygame.Surface((3, 3), 0, 32)
    ar = pygame.PixelArray(sf)
    ar[:] = (20, 30, 40)
    self.assertEqual(ar[0, 0], sf.map_rgb((20, 30, 40)))
    ar[:] = [20, 30, 40]
    self.assertEqual(ar[0, 0], 20)
    self.assertEqual(ar[1, 0], 30)
    self.assertEqual(ar[2, 0], 40)
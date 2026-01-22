import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_length_1_dimension_broadcast(self):
    w = 5
    sf = pygame.Surface((w, w), 0, 32)
    ar = pygame.PixelArray(sf)
    sf_x = pygame.Surface((w, 1), 0, 32)
    ar_x = pygame.PixelArray(sf_x)
    for i in range(w):
        ar_x[i, 0] = (w + 1) * 10
    ar[...] = ar_x
    for y in range(w):
        for x in range(w):
            self.assertEqual(ar[x, y], ar_x[x, 0])
    ar[...] = 0
    sf_y = pygame.Surface((1, w), 0, 32)
    ar_y = pygame.PixelArray(sf_y)
    for i in range(w):
        ar_y[0, i] = (w + 1) * 10
    ar[...] = ar_y
    for x in range(w):
        for y in range(w):
            self.assertEqual(ar[x, y], ar_y[0, y])
    ar[...] = 0
    sf_1px = pygame.Surface((1, 1), 0, 32)
    ar_1px = pygame.PixelArray(sf_1px)
    ar_1px[0, 0] = 42
    ar[...] = ar_1px
    for y in range(w):
        for x in range(w):
            self.assertEqual(ar[x, y], 42)
import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_self_assign(self):
    w = 10
    max_x = w - 1
    h = 20
    max_y = h - 1
    for bpp in [1, 2, 3, 4]:
        sf = pygame.Surface((w, h), 0, bpp * 8)
        ar = pygame.PixelArray(sf)
        for i in range(w * h):
            ar[i % w, i // w] = i
        ar[:, :] = ar[::-1, :]
        for i in range(w * h):
            self.assertEqual(ar[max_x - i % w, i // w], i)
        ar = pygame.PixelArray(sf)
        for i in range(w * h):
            ar[i % w, i // w] = i
        ar[:, :] = ar[:, ::-1]
        for i in range(w * h):
            self.assertEqual(ar[i % w, max_y - i // w], i)
        ar = pygame.PixelArray(sf)
        for i in range(w * h):
            ar[i % w, i // w] = i
        ar[:, :] = ar[::-1, ::-1]
        for i in range(w * h):
            self.assertEqual(ar[max_x - i % w, max_y - i // w], i)
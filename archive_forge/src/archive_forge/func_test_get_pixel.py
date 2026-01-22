import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
@unittest.skipIf(IS_PYPY, 'pypy malloc abort')
def test_get_pixel(self):
    w = 10
    h = 20
    size = (w, h)
    bg_color = (0, 0, 255)
    fg_color_y = (0, 0, 128)
    fg_color_x = (0, 0, 11)
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface(size, 0, bpp)
        mapped_bg_color = sf.map_rgb(bg_color)
        mapped_fg_color_y = sf.map_rgb(fg_color_y)
        mapped_fg_color_x = sf.map_rgb(fg_color_x)
        self.assertNotEqual(mapped_fg_color_y, mapped_bg_color, 'Unusable test colors for bpp %i' % (bpp,))
        self.assertNotEqual(mapped_fg_color_x, mapped_bg_color, 'Unusable test colors for bpp %i' % (bpp,))
        self.assertNotEqual(mapped_fg_color_y, mapped_fg_color_x, 'Unusable test colors for bpp %i' % (bpp,))
        sf.fill(bg_color)
        ar = pygame.PixelArray(sf)
        ar_y = ar.__getitem__(1)
        for y in range(h):
            ar2 = ar_y.__getitem__(y)
            self.assertEqual(ar2, mapped_bg_color, 'ar[1][%i] == %i, mapped_bg_color == %i' % (y, ar2, mapped_bg_color))
            sf.set_at((1, y), fg_color_y)
            ar2 = ar_y.__getitem__(y)
            self.assertEqual(ar2, mapped_fg_color_y, 'ar[1][%i] == %i, mapped_fg_color_y == %i' % (y, ar2, mapped_fg_color_y))
        sf.set_at((1, 1), bg_color)
        for x in range(w):
            ar2 = ar.__getitem__(x).__getitem__(1)
            self.assertEqual(ar2, mapped_bg_color, 'ar[%i][1] = %i, mapped_bg_color = %i' % (x, ar2, mapped_bg_color))
            sf.set_at((x, 1), fg_color_x)
            ar2 = ar.__getitem__(x).__getitem__(1)
            self.assertEqual(ar2, mapped_fg_color_x, 'ar[%i][1] = %i, mapped_fg_color_x = %i' % (x, ar2, mapped_fg_color_x))
        ar2 = ar.__getitem__(0).__getitem__(0)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(1).__getitem__(0)
        self.assertEqual(ar2, mapped_fg_color_y, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(-4).__getitem__(1)
        self.assertEqual(ar2, mapped_fg_color_x, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(-4).__getitem__(5)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(-4).__getitem__(0)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(-w + 1).__getitem__(0)
        self.assertEqual(ar2, mapped_fg_color_y, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(-w).__getitem__(0)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(5).__getitem__(-4)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(5).__getitem__(-h + 1)
        self.assertEqual(ar2, mapped_fg_color_x, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(5).__getitem__(-h)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(0).__getitem__(-h + 1)
        self.assertEqual(ar2, mapped_fg_color_x, 'bpp = %i' % (bpp,))
        ar2 = ar.__getitem__(0).__getitem__(-h)
        self.assertEqual(ar2, mapped_bg_color, 'bpp = %i' % (bpp,))
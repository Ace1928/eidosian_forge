import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_as_class(self):
    sf = pygame.Surface((2, 3), 0, 32)
    ar = pygame.PixelArray(sf)
    self.assertRaises(AttributeError, getattr, ar, 'nonnative')
    ar.nonnative = 'value'
    self.assertEqual(ar.nonnative, 'value')
    r = weakref.ref(ar)
    self.assertTrue(r() is ar)
    del ar
    gc.collect()
    self.assertTrue(r() is None)

    class C(pygame.PixelArray):

        def __str__(self):
            return 'string (%i, %i)' % self.shape
    ar = C(sf)
    self.assertEqual(str(ar), 'string (2, 3)')
    r = weakref.ref(ar)
    self.assertTrue(r() is ar)
    del ar
    gc.collect()
    self.assertTrue(r() is None)
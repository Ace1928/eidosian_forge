import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def assert_surfaces_equal(self, s1, s2, msg=None):
    """Checks if two surfaces are equal in size and color."""
    w, h = s1.get_size()
    self.assertTupleEqual((w, h), s2.get_size(), msg)
    msg = '' if msg is None else f'{msg}, '
    msg += f'size: ({w}, {h})'
    for x in range(w):
        for y in range(h):
            self.assertEqual(s1.get_at((x, y)), s2.get_at((x, y)), f'{msg}, position: ({x}, {y})')
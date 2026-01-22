import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def assert_surface_filled(self, surface, expected_color, msg=None):
    """Checks if the surface is filled with the given color."""
    width, height = surface.get_size()
    surface.lock()
    for pos in ((x, y) for y in range(height) for x in range(width)):
        self.assertEqual(surface.get_at(pos), expected_color, msg)
    surface.unlock()
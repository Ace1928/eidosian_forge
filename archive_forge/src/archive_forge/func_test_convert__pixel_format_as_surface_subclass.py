import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_convert__pixel_format_as_surface_subclass(self):
    """Ensure convert accepts a Surface subclass argument."""
    expected_size = (23, 17)
    convert_surface = SurfaceSubclass(expected_size, 0, 32)
    depth_surface = SurfaceSubclass((31, 61), 0, 32)
    pygame.display.init()
    try:
        surface = convert_surface.convert(depth_surface)
        self.assertIsNot(surface, depth_surface)
        self.assertIsNot(surface, convert_surface)
        self.assertIsInstance(surface, pygame.Surface)
        self.assertIsInstance(surface, SurfaceSubclass)
        self.assertEqual(surface.get_size(), expected_size)
    finally:
        pygame.display.quit()
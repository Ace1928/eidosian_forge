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
def test_blit_area_contraint(self):
    """Testing area constraint"""
    result = self.dst_surface.blit(self.src_surface, dest=pygame.Rect((1, 1, 1, 1)), area=pygame.Rect((2, 2, 2, 2)))
    self.assertIsInstance(result, pygame.Rect)
    self.assertEqual(result.size, (2, 2))
    self.assertEqual(self.dst_surface.get_at((0, 0)), (0, 0, 0))
    self.assertEqual(self.dst_surface.get_at((63, 0)), (0, 0, 0))
    self.assertEqual(self.dst_surface.get_at((0, 63)), (0, 0, 0))
    self.assertEqual(self.dst_surface.get_at((63, 63)), (0, 0, 0))
    self.assertEqual(self.dst_surface.get_at((1, 1)), (255, 255, 255))
    self.assertEqual(self.dst_surface.get_at((2, 2)), (255, 255, 255))
    self.assertEqual(self.dst_surface.get_at((3, 3)), (0, 0, 0))
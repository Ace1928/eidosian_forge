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
def test_map_rgb(self):
    color = Color(0, 128, 255, 64)
    surf = pygame.Surface((5, 5), SRCALPHA, 32)
    c = surf.map_rgb(color)
    self.assertEqual(surf.unmap_rgb(c), color)
    self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 0))
    surf.fill(c)
    self.assertEqual(surf.get_at((0, 0)), color)
    surf.fill((0, 0, 0, 0))
    self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 0))
    surf.set_at((0, 0), c)
    self.assertEqual(surf.get_at((0, 0)), color)
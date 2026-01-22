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
def test_set_palette__set_at(self):
    surf = pygame.Surface((2, 2), depth=8)
    palette = 256 * [(10, 20, 30)]
    palette[1] = (50, 40, 30)
    surf.set_palette(palette)
    surf.set_at((0, 0), (60, 50, 40))
    self.assertEqual(surf.get_at((0, 0)), (50, 40, 30, 255))
    self.assertEqual(surf.get_at((1, 0)), (10, 20, 30, 255))
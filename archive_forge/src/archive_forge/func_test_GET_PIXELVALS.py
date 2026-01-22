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
def test_GET_PIXELVALS(self):
    src = self._make_surface(32, srcalpha=True)
    src.fill((0, 0, 0, 128))
    src.set_alpha(None)
    dst = self._make_surface(32, srcalpha=True)
    dst.blit(src, (0, 0), special_flags=BLEND_RGBA_ADD)
    self.assertEqual(dst.get_at((0, 0)), (0, 0, 0, 255))
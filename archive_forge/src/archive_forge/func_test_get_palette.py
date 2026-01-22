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
def test_get_palette(self):
    palette = [Color(i, i, i) for i in range(256)]
    surf = pygame.Surface((2, 2), 0, 8)
    surf.set_palette(palette)
    palette2 = surf.get_palette()
    self.assertEqual(len(palette2), len(palette))
    for c2, c in zip(palette2, palette):
        self.assertEqual(c2, c)
    for c in palette2:
        self.assertIsInstance(c, pygame.Color)
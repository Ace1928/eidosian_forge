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
def test_set_palette__fail(self):
    palette = 256 * [(10, 20, 30)]
    surf = pygame.Surface((2, 2), 0, 32)
    self.assertRaises(pygame.error, surf.set_palette, palette)
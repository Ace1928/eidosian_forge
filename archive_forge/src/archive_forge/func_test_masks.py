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
def test_masks(self):

    def make_surf(bpp, flags, masks):
        pygame.Surface((10, 10), flags, bpp, masks)
    masks = (4278190080, 16711680, 65280, 0)
    self.assertEqual(make_surf(32, 0, masks), None)
    masks = (8323072, 65280, 255, 0)
    self.assertRaises(ValueError, make_surf, 24, 0, masks)
    self.assertRaises(ValueError, make_surf, 32, 0, masks)
    masks = (7274496, 65280, 255, 0)
    self.assertRaises(ValueError, make_surf, 32, 0, masks)
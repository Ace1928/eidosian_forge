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
def test_get_masks__rgb(self):
    """
        Ensure that get_mask can return RGB mask.
        """
    masks = [(96, 28, 3, 0), (3840, 240, 15, 0), (31744, 992, 31, 0), (63488, 2016, 31, 0), (16711680, 65280, 255, 0), (16711680, 65280, 255, 0)]
    depths = [8, 12, 15, 16, 24, 32]
    for expected, depth in list(zip(masks, depths)):
        surface = pygame.Surface((10, 10), 0, depth)
        if depth == 8:
            expected = (0, 0, 0, 0)
        self.assertEqual(expected, surface.get_masks())
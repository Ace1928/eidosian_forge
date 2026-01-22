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
def test_get_masks__rgba(self):
    """
        Ensure that get_mask can return RGBA mask.
        """
    masks = [(3840, 240, 15, 61440), (16711680, 65280, 255, 4278190080)]
    depths = [16, 32]
    for expected, depth in list(zip(masks, depths)):
        surface = pygame.Surface((10, 10), pygame.SRCALPHA, depth)
        self.assertEqual(expected, surface.get_masks())
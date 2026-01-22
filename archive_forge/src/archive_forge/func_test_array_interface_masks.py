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
def test_array_interface_masks(self):
    """Test non-default color byte orders on 3D views"""
    sz = (5, 7)
    s = pygame.Surface(sz, 0, 32)
    s_masks = list(s.get_masks())
    masks = [255, 65280, 16711680]
    if s_masks[0:3] == masks or s_masks[0:3] == masks[::-1]:
        masks = s_masks[2::-1] + s_masks[3:4]
        self._check_interface_3D(pygame.Surface(sz, 0, 32, masks))
    s = pygame.Surface(sz, 0, 24)
    s_masks = list(s.get_masks())
    masks = [255, 65280, 16711680]
    if s_masks[0:3] == masks or s_masks[0:3] == masks[::-1]:
        masks = s_masks[2::-1] + s_masks[3:4]
        self._check_interface_3D(pygame.Surface(sz, 0, 24, masks))
    masks = [65280, 16711680, 4278190080, 0]
    self._check_interface_3D(pygame.Surface(sz, 0, 32, masks))
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
def test_array_interface_rgb(self):
    for shifts in [[0, 8, 16, 24], [8, 16, 24, 0], [24, 16, 8, 0], [16, 8, 0, 24]]:
        masks = [255 << s for s in shifts]
        masks[3] = 0
        for plane in range(3):
            s = pygame.Surface((4, 2), 0, 24)
            self._check_interface_rgba(s, plane)
            s = pygame.Surface((4, 2), 0, 32)
            self._check_interface_rgba(s, plane)
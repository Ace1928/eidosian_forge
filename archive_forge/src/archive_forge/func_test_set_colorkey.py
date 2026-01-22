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
def test_set_colorkey(self):
    s = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
    colorkeys = ((20, 189, 20, 255), (128, 50, 50, 255), (23, 21, 255, 255))
    for colorkey in colorkeys:
        s.set_colorkey(colorkey)
        for t in range(4):
            s.set_colorkey(s.get_colorkey())
        self.assertEqual(s.get_colorkey(), colorkey)
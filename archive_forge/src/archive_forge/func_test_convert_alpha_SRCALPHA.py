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
def test_convert_alpha_SRCALPHA(self):
    """Ensure that the surface returned by surf.convert_alpha()
        has alpha blending enabled"""
    pygame.display.init()
    try:
        pygame.display.set_mode((640, 480))
        s1 = pygame.Surface((100, 100), 0, 32)
        s1_alpha = s1.convert_alpha()
        self.assertEqual(s1_alpha.get_flags() & SRCALPHA, SRCALPHA)
        self.assertEqual(s1_alpha.get_alpha(), 255)
    finally:
        pygame.display.quit()
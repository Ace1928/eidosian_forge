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
def test_flags_default0_display(self):
    """is set to zero, and SRCALPH is not set by default even when the display is initialized."""
    pygame.display.set_mode((320, 200))
    try:
        surf = pygame.Surface((70, 70))
        self.assertEqual(surf.get_flags() & SRCALPHA, 0)
    finally:
        pygame.display.quit()
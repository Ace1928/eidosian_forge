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
def test_get_flags(self):
    """Ensure a surface's flags can be retrieved."""
    s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
    self.assertEqual(s1.get_flags(), pygame.SRCALPHA)
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
def test_get_clip(self):
    s = pygame.Surface((800, 600))
    rectangle = s.get_clip()
    self.assertEqual(rectangle, (0, 0, 800, 600))
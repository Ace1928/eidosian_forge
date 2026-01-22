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
def test_surface_created_opaque_black(self):
    surf = pygame.Surface((20, 20))
    self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 255))
    pygame.display.set_mode((500, 500))
    surf = pygame.Surface((20, 20))
    self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 255))
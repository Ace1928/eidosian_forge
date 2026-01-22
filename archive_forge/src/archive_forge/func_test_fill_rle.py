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
def test_fill_rle(self):
    """Test RLEACCEL flag with fill()"""
    color = (250, 25, 25, 255)
    surf = pygame.Surface((32, 32))
    blit_surf = pygame.Surface((32, 32))
    blit_surf.set_colorkey((255, 0, 255), pygame.RLEACCEL)
    self.assertTrue(blit_surf.get_flags() & pygame.RLEACCELOK)
    surf.blit(blit_surf, (0, 0))
    blit_surf.fill(color)
    self.assertEqual(blit_surf.mustlock(), blit_surf.get_flags() & pygame.RLEACCEL != 0)
    self.assertTrue(blit_surf.get_flags() & pygame.RLEACCEL)
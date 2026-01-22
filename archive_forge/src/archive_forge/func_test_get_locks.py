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
def test_get_locks(self):
    surface = pygame.Surface((100, 100))
    self.assertEqual(surface.get_locks(), ())
    surface.lock()
    self.assertEqual(surface.get_locks(), (surface,))
    surface.unlock()
    self.assertEqual(surface.get_locks(), ())
    pxarray = pygame.PixelArray(surface)
    self.assertNotEqual(surface.get_locks(), ())
    pxarray.close()
    self.assertEqual(surface.get_locks(), ())
    with self.assertRaises(AttributeError):
        'DUMMY'.get_locks()
    surface.lock()
    surface.lock()
    surface.lock()
    self.assertEqual(surface.get_locks(), (surface, surface, surface))
    surface.unlock()
    surface.unlock()
    self.assertEqual(surface.get_locks(), (surface,))
    surface.unlock()
    self.assertEqual(surface.get_locks(), ())
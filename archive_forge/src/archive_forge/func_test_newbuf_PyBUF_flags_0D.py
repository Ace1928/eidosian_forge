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
@unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
def test_newbuf_PyBUF_flags_0D(self):
    from pygame.tests.test_utils import buftools
    Importer = buftools.Importer
    s = pygame.Surface((10, 6), 0, 32)
    a = s.get_view('0')
    b = Importer(a, buftools.PyBUF_SIMPLE)
    self.assertEqual(b.ndim, 0)
    self.assertTrue(b.format is None)
    self.assertEqual(b.len, a.length)
    self.assertEqual(b.itemsize, 1)
    self.assertTrue(b.shape is None)
    self.assertTrue(b.strides is None)
    self.assertTrue(b.suboffsets is None)
    self.assertFalse(b.readonly)
    self.assertEqual(b.buf, s._pixels_address)
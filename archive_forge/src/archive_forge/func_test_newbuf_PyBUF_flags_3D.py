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
def test_newbuf_PyBUF_flags_3D(self):
    from pygame.tests.test_utils import buftools
    Importer = buftools.Importer
    s = pygame.Surface((12, 6), 0, 24)
    rmask, gmask, bmask, amask = s.get_masks()
    if self.lilendian:
        if rmask == 255:
            color_step = 1
            addr_offset = 0
        else:
            color_step = -1
            addr_offset = 2
    elif rmask == 16711680:
        color_step = 1
        addr_offset = 0
    else:
        color_step = -1
        addr_offset = 2
    a = s.get_view('3')
    b = Importer(a, buftools.PyBUF_STRIDES)
    w, h = s.get_size()
    shape = (w, h, 3)
    strides = (3, s.get_pitch(), color_step)
    self.assertEqual(b.ndim, 3)
    self.assertTrue(b.format is None)
    self.assertEqual(b.len, a.length)
    self.assertEqual(b.itemsize, 1)
    self.assertEqual(b.shape, shape)
    self.assertEqual(b.strides, strides)
    self.assertTrue(b.suboffsets is None)
    self.assertFalse(b.readonly)
    self.assertEqual(b.buf, s._pixels_address + addr_offset)
    b = Importer(a, buftools.PyBUF_RECORDS_RO)
    self.assertEqual(b.ndim, 3)
    self.assertEqual(b.format, 'B')
    self.assertEqual(b.strides, strides)
    b = Importer(a, buftools.PyBUF_RECORDS)
    self.assertEqual(b.ndim, 3)
    self.assertEqual(b.format, 'B')
    self.assertEqual(b.strides, strides)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)
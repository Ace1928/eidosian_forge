import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_newbuf_1D(self):
    buftools = self.buftools
    Importer = buftools.Importer
    s = pygame.Surface((2, 16), 0, 32)
    ar_2D = pygame.PixelArray(s)
    x = 0
    ar = ar_2D[x]
    format = self.bitsize_to_format[s.get_bitsize()]
    itemsize = ar.itemsize
    shape = ar.shape
    h = shape[0]
    strides = ar.strides
    length = h * itemsize
    buf = s._pixels_address + x * itemsize
    imp = Importer(ar, buftools.PyBUF_STRIDES)
    self.assertTrue(imp.obj, ar)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 1)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertTrue(imp.suboffsets is None)
    self.assertEqual(imp.buf, buf)
    imp = Importer(ar, buftools.PyBUF_FULL)
    self.assertEqual(imp.ndim, 1)
    self.assertEqual(imp.format, format)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_SIMPLE)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_FORMAT)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_WRITABLE)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_ND)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_C_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_F_CONTIGUOUS)
    self.assertRaises(BufferError, Importer, ar, buftools.PyBUF_ANY_CONTIGUOUS)
    y = 10
    ar = ar_2D[:, y]
    shape = ar.shape
    w = shape[0]
    strides = ar.strides
    length = w * itemsize
    buf = s._pixels_address + y * s.get_pitch()
    imp = Importer(ar, buftools.PyBUF_FULL)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 1)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertEqual(imp.format, format)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertEqual(imp.buf, buf)
    self.assertTrue(imp.suboffsets is None)
    imp = Importer(ar, buftools.PyBUF_SIMPLE)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 0)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertTrue(imp.shape is None)
    self.assertTrue(imp.strides is None)
    imp = Importer(ar, buftools.PyBUF_ND)
    self.assertEqual(imp.len, length)
    self.assertEqual(imp.ndim, 1)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.format is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.shape, shape)
    self.assertTrue(imp.strides is None)
    imp = Importer(ar, buftools.PyBUF_C_CONTIGUOUS)
    self.assertEqual(imp.ndim, 1)
    imp = Importer(ar, buftools.PyBUF_F_CONTIGUOUS)
    self.assertEqual(imp.ndim, 1)
    imp = Importer(ar, buftools.PyBUF_ANY_CONTIGUOUS)
    self.assertEqual(imp.ndim, 1)
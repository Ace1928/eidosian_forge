import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_getslice_cint(self):
    a = (c_int * 100)(*range(1100, 1200))
    b = list(range(1100, 1200))
    self.assertEqual(a[0:2], b[0:2])
    self.assertEqual(a[0:2], b[0:2])
    self.assertEqual(len(a), len(b))
    self.assertEqual(a[5:7], b[5:7])
    self.assertEqual(a[5:7], b[5:7])
    self.assertEqual(a[-1], b[-1])
    self.assertEqual(a[:], b[:])
    self.assertEqual(a[:], b[:])
    self.assertEqual(a[10::-1], b[10::-1])
    self.assertEqual(a[30:20:-1], b[30:20:-1])
    self.assertEqual(a[:12:6], b[:12:6])
    self.assertEqual(a[2:6:4], b[2:6:4])
    a[0:5] = range(5, 10)
    self.assertEqual(a[0:5], list(range(5, 10)))
    self.assertEqual(a[0:5], list(range(5, 10)))
    self.assertEqual(a[4::-1], list(range(9, 4, -1)))
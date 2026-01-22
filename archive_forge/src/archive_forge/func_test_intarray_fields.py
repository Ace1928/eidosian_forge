import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
def test_intarray_fields(self):

    class SomeInts(Structure):
        _fields_ = [('a', c_int * 4)]
    self.assertEqual(SomeInts((1, 2)).a[:], [1, 2, 0, 0])
    self.assertEqual(SomeInts((1, 2)).a[:], [1, 2, 0, 0])
    self.assertEqual(SomeInts((1, 2)).a[::-1], [0, 0, 2, 1])
    self.assertEqual(SomeInts((1, 2)).a[::2], [1, 0])
    self.assertEqual(SomeInts((1, 2)).a[1:5:6], [2])
    self.assertEqual(SomeInts((1, 2)).a[6:4:-1], [])
    self.assertEqual(SomeInts((1, 2, 3, 4)).a[:], [1, 2, 3, 4])
    self.assertEqual(SomeInts((1, 2, 3, 4)).a[:], [1, 2, 3, 4])
    self.assertRaises(RuntimeError, SomeInts, (1, 2, 3, 4, 5))
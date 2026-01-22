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
def test_none_to_pointer_fields(self):

    class S(Structure):
        _fields_ = [('x', c_int), ('p', POINTER(c_int))]
    s = S()
    s.x = 12345678
    s.p = None
    self.assertEqual(s.x, 12345678)
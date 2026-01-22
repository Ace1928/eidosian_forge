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
def test_keyword_initializers(self):

    class POINT(Structure):
        _fields_ = [('x', c_int), ('y', c_int)]
    pt = POINT(1, 2)
    self.assertEqual((pt.x, pt.y), (1, 2))
    pt = POINT(y=2, x=1)
    self.assertEqual((pt.x, pt.y), (1, 2))
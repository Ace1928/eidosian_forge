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
@unittest.skipIf(_architecture() == ('64bit', 'WindowsPE'), "can't test Windows x64 build")
@unittest.skipUnless(sys.byteorder == 'little', "can't test on this platform")
def test_issue18060_b(self):

    class Base(Structure):
        _fields_ = [('y', c_double), ('x', c_double)]

    class Mid(Base):
        _fields_ = []

    class Vector(Mid):
        _fields_ = []
    self._test_issue18060(Vector)
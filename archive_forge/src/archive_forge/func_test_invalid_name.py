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
def test_invalid_name(self):

    def declare_with_name(name):

        class S(Structure):
            _fields_ = [(name, c_int)]
    self.assertRaises(TypeError, declare_with_name, b'x')
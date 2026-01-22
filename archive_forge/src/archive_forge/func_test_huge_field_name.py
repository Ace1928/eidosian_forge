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
def test_huge_field_name(self):

    def create_class(length):

        class S(Structure):
            _fields_ = [('x' * length, c_int)]
    for length in [10 ** i for i in range(0, 8)]:
        try:
            create_class(length)
        except MemoryError:
            pass
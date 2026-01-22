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
def test_contains_itself(self):

    class Recursive(Structure):
        pass
    try:
        Recursive._fields_ = [('next', Recursive)]
    except AttributeError as details:
        self.assertIn('Structure or union cannot contain itself', str(details))
    else:
        self.fail('Structure or union cannot contain itself')
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
def test_pass_by_value_finalizer(self):
    finalizer_calls = []

    class Test(Structure):
        _fields_ = [('first', c_ulong), ('second', c_ulong), ('third', c_ulong)]

        def __del__(self):
            finalizer_calls.append('called')
    s = Test(1, 2, 3)
    self.assertGreater(sizeof(s), sizeof(c_void_p))
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_large_struct_update_value
    func.argtypes = (Test,)
    func.restype = None
    func(s)
    self.assertEqual(finalizer_calls, [])
    self.assertEqual(s.first, 1)
    self.assertEqual(s.second, 2)
    self.assertEqual(s.third, 3)
    s = None
    support.gc_collect()
    self.assertEqual(finalizer_calls, ['called'])
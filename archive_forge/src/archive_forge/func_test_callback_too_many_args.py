import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_callback_too_many_args(self):

    def func(*args):
        return len(args)
    proto = CFUNCTYPE(c_int, *(c_int,) * CTYPES_MAX_ARGCOUNT)
    cb = proto(func)
    args1 = (1,) * CTYPES_MAX_ARGCOUNT
    self.assertEqual(cb(*args1), CTYPES_MAX_ARGCOUNT)
    args2 = (1,) * (CTYPES_MAX_ARGCOUNT + 1)
    with self.assertRaises(ArgumentError):
        cb(*args2)
    with self.assertRaises(ArgumentError):
        CFUNCTYPE(c_int, *(c_int,) * (CTYPES_MAX_ARGCOUNT + 1))
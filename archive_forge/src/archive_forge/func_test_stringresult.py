from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_stringresult(self):
    f = dll._testfunc_p_p
    f.argtypes = None
    f.restype = c_char_p
    result = f(b'123')
    self.assertEqual(result, b'123')
    result = f(None)
    self.assertEqual(result, None)
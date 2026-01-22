from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_doubleresult(self):
    f = dll._testfunc_d_bhilfd
    f.argtypes = [c_byte, c_short, c_int, c_long, c_float, c_double]
    f.restype = c_double
    result = f(1, 2, 3, 4, 5.0, 6.0)
    self.assertEqual(result, 21)
    self.assertEqual(type(result), float)
    result = f(-1, -2, -3, -4, -5.0, -6.0)
    self.assertEqual(result, -21)
    self.assertEqual(type(result), float)
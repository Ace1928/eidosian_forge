from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@need_symbol('c_wchar')
def test_wchar_result(self):
    f = dll._testfunc_i_bhilfd
    f.argtypes = [c_byte, c_short, c_int, c_long, c_float, c_double]
    f.restype = c_wchar
    result = f(0, 0, 0, 0, 0, 0)
    self.assertEqual(result, '\x00')
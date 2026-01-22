from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@need_symbol('c_wchar')
def test_wchar_parm(self):
    f = dll._testfunc_i_bhilfd
    f.argtypes = [c_byte, c_wchar, c_int, c_long, c_float, c_double]
    result = f(1, 'x', 3, 4, 5.0, 6.0)
    self.assertEqual(result, 139)
    self.assertEqual(type(result), int)
from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@need_symbol('c_longlong')
def test_longlongresult(self):
    f = dll._testfunc_q_bhilfd
    f.restype = c_longlong
    f.argtypes = [c_byte, c_short, c_int, c_long, c_float, c_double]
    result = f(1, 2, 3, 4, 5.0, 6.0)
    self.assertEqual(result, 21)
    f = dll._testfunc_q_bhilfdq
    f.restype = c_longlong
    f.argtypes = [c_byte, c_short, c_int, c_long, c_float, c_double, c_longlong]
    result = f(1, 2, 3, 4, 5.0, 6.0, 21)
    self.assertEqual(result, 42)
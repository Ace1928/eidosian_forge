from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@need_symbol('c_longlong')
def test_longlong_callbacks(self):
    f = dll._testfunc_callback_q_qf
    f.restype = c_longlong
    MyCallback = CFUNCTYPE(c_longlong, c_longlong)
    f.argtypes = [c_longlong, MyCallback]

    def callback(value):
        self.assertIsInstance(value, int)
        return value & 2147483647
    cb = MyCallback(callback)
    self.assertEqual(13577625587, f(1000000000000, cb))
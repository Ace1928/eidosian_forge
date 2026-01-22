from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_callbacks_2(self):
    f = dll._testfunc_callback_i_if
    f.restype = c_int
    MyCallback = CFUNCTYPE(c_int, c_int)
    f.argtypes = [c_int, MyCallback]

    def callback(value):
        self.assertEqual(type(value), int)
        return value
    cb = MyCallback(callback)
    result = f(-10, cb)
    self.assertEqual(result, -18)
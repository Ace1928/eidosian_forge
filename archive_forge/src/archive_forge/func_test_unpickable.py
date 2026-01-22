import unittest
import pickle
from ctypes import *
import _ctypes_test
def test_unpickable(self):
    self.assertRaises(ValueError, lambda: self.dumps(Y()))
    prototype = CFUNCTYPE(c_int)
    for item in [c_char_p(), c_wchar_p(), c_void_p(), pointer(c_int(42)), dll._testfunc_p_p, prototype(lambda: 42)]:
        self.assertRaises(ValueError, lambda: self.dumps(item))
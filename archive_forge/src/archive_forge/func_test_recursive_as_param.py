import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_recursive_as_param(self):
    from ctypes import c_int

    class A:
        pass
    a = A()
    a._as_parameter_ = a
    with self.assertRaises(RecursionError):
        c_int.from_param(a)
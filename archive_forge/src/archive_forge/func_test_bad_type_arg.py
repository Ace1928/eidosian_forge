from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
def test_bad_type_arg(self):
    array_type = c_byte * sizeof(c_int)
    array = array_type()
    self.assertRaises(TypeError, cast, array, None)
    self.assertRaises(TypeError, cast, array, array_type)

    class Struct(Structure):
        _fields_ = [('a', c_int)]
    self.assertRaises(TypeError, cast, array, Struct)

    class MyUnion(Union):
        _fields_ = [('a', c_int)]
    self.assertRaises(TypeError, cast, array, MyUnion)
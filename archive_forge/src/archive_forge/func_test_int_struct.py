import unittest
from ctypes import *
def test_int_struct(self):

    class X(Structure):
        _fields_ = [('x', MyInt)]
    self.assertEqual(X().x, MyInt())
    s = X()
    s.x = MyInt(42)
    self.assertEqual(s.x, MyInt(42))
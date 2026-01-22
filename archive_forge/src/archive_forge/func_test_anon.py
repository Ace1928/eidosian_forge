import unittest
import test.support
from ctypes import *
def test_anon(self):

    class ANON(Union):
        _fields_ = [('a', c_int), ('b', c_int)]

    class Y(Structure):
        _fields_ = [('x', c_int), ('_', ANON), ('y', c_int)]
        _anonymous_ = ['_']
    self.assertEqual(Y.a.offset, sizeof(c_int))
    self.assertEqual(Y.b.offset, sizeof(c_int))
    self.assertEqual(ANON.a.offset, 0)
    self.assertEqual(ANON.b.offset, 0)
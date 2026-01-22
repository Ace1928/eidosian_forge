from ctypes import *
import unittest
def test_ccharp_struct(self):

    class X(Structure):
        _fields_ = [('a', c_char_p), ('b', c_char_p)]
    x = X()
    self.assertEqual(x._objects, None)
    x.a = b'spam'
    x.b = b'foo'
    self.assertEqual(x._objects, {'0': b'spam', '1': b'foo'})
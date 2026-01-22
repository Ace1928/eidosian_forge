import unittest
from ctypes import *
def test_5(self):

    class X(Structure):
        _fields_ = (('char', c_char * 5),)
    x = X(b'#' * 5)
    x.char = b'a\x00b\x00'
    self.assertEqual(bytes(x), b'a\x00###')
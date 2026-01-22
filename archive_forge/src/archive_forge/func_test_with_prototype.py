import unittest
from ctypes import *
import _ctypes_test
def test_with_prototype(self):
    dll = CDLL(_ctypes_test.__file__)
    get_strchr = dll.get_strchr
    get_strchr.restype = CFUNCTYPE(c_char_p, c_char_p, c_char)
    strchr = get_strchr()
    self.assertEqual(strchr(b'abcdef', b'b'), b'bcdef')
    self.assertEqual(strchr(b'abcdef', b'x'), None)
    self.assertEqual(strchr(b'abcdef', 98), b'bcdef')
    self.assertEqual(strchr(b'abcdef', 107), None)
    self.assertRaises(ArgumentError, strchr, b'abcdef', 3.0)
    self.assertRaises(TypeError, strchr, b'abcdef')
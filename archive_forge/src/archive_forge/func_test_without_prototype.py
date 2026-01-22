import unittest
from ctypes import *
import _ctypes_test
def test_without_prototype(self):
    dll = CDLL(_ctypes_test.__file__)
    get_strchr = dll.get_strchr
    get_strchr.restype = c_void_p
    addr = get_strchr()
    strchr = CFUNCTYPE(c_char_p, c_char_p, c_char)(addr)
    self.assertTrue(strchr(b'abcdef', b'b'), 'bcdef')
    self.assertEqual(strchr(b'abcdef', b'x'), None)
    self.assertRaises(ArgumentError, strchr, b'abcdef', 3.0)
    self.assertRaises(TypeError, strchr, b'abcdef')
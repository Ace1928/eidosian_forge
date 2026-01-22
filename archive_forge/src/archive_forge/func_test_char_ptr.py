import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_char_ptr(self):
    s = b'abcdefghijklmnopqrstuvwxyz'
    dll = CDLL(_ctypes_test.__file__)
    dll.my_strdup.restype = POINTER(c_char)
    dll.my_free.restype = None
    res = dll.my_strdup(s)
    self.assertEqual(res[:len(s)], s)
    self.assertEqual(res[:3], s[:3])
    self.assertEqual(res[:len(s)], s)
    self.assertEqual(res[len(s) - 1:-1:-1], s[::-1])
    self.assertEqual(res[len(s) - 1:5:-7], s[:5:-7])
    self.assertEqual(res[0:-1:-1], s[0::-1])
    import operator
    self.assertRaises(ValueError, operator.getitem, res, slice(None, None, None))
    self.assertRaises(ValueError, operator.getitem, res, slice(0, None, None))
    self.assertRaises(ValueError, operator.getitem, res, slice(None, 5, -1))
    self.assertRaises(ValueError, operator.getitem, res, slice(-5, None, None))
    self.assertRaises(TypeError, operator.setitem, res, slice(0, 5), 'abcde')
    dll.my_free(res)
    dll.my_strdup.restype = POINTER(c_byte)
    res = dll.my_strdup(s)
    self.assertEqual(res[:len(s)], list(range(ord('a'), ord('z') + 1)))
    self.assertEqual(res[:len(s)], list(range(ord('a'), ord('z') + 1)))
    dll.my_free(res)
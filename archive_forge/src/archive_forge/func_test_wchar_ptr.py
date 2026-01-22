import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
@need_symbol('c_wchar')
def test_wchar_ptr(self):
    s = 'abcdefghijklmnopqrstuvwxyz\x00'
    dll = CDLL(_ctypes_test.__file__)
    dll.my_wcsdup.restype = POINTER(c_wchar)
    dll.my_wcsdup.argtypes = (POINTER(c_wchar),)
    dll.my_free.restype = None
    res = dll.my_wcsdup(s[:-1])
    self.assertEqual(res[:len(s)], s)
    self.assertEqual(res[:len(s)], s)
    self.assertEqual(res[len(s) - 1:-1:-1], s[::-1])
    self.assertEqual(res[len(s) - 1:5:-7], s[:5:-7])
    import operator
    self.assertRaises(TypeError, operator.setitem, res, slice(0, 5), 'abcde')
    dll.my_free(res)
    if sizeof(c_wchar) == sizeof(c_short):
        dll.my_wcsdup.restype = POINTER(c_short)
    elif sizeof(c_wchar) == sizeof(c_int):
        dll.my_wcsdup.restype = POINTER(c_int)
    elif sizeof(c_wchar) == sizeof(c_long):
        dll.my_wcsdup.restype = POINTER(c_long)
    else:
        self.skipTest('Pointers to c_wchar are not supported')
    res = dll.my_wcsdup(s[:-1])
    tmpl = list(range(ord('a'), ord('z') + 1))
    self.assertEqual(res[:len(s) - 1], tmpl)
    self.assertEqual(res[:len(s) - 1], tmpl)
    self.assertEqual(res[len(s) - 2:-1:-1], tmpl[::-1])
    self.assertEqual(res[len(s) - 2:5:-7], tmpl[:5:-7])
    dll.my_free(res)
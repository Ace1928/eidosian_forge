import unittest, sys
from ctypes import *
import _ctypes_test
def test_change_pointers(self):
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_p_p
    i = c_int(89284)
    func.restype = POINTER(c_int)
    func.argtypes = (POINTER(c_int),)
    res = func(pointer(i))
    self.assertEqual(res[0], 89284)
    self.assertEqual(res.contents.value, 89284)
    res[0] = 54345
    self.assertEqual(i.value, 54345)
    x = c_int(12321)
    res.contents = x
    self.assertEqual(i.value, 54345)
    x.value = -99
    self.assertEqual(res.contents.value, -99)
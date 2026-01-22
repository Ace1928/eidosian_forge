from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
def test_POINTER_c_char_arg(self):
    func = testdll._testfunc_p_p
    func.restype = c_char_p
    func.argtypes = (POINTER(c_char),)
    self.assertEqual(None, func(None))
    self.assertEqual(b'123', func(b'123'))
    self.assertEqual(None, func(c_char_p(None)))
    self.assertEqual(b'123', func(c_char_p(b'123')))
    self.assertEqual(b'123', func(c_buffer(b'123')))
    ca = c_char(b'a')
    self.assertEqual(ord(b'a'), func(pointer(ca))[0])
    self.assertEqual(ord(b'a'), func(byref(ca))[0])
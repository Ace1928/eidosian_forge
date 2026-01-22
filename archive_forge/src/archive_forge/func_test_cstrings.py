import unittest
from ctypes.test import need_symbol
import test.support
def test_cstrings(self):
    from ctypes import c_char_p
    s = b'123'
    self.assertIs(c_char_p.from_param(s)._obj, s)
    self.assertEqual(c_char_p.from_param(b'123')._obj, b'123')
    self.assertRaises(TypeError, c_char_p.from_param, '123ÿ')
    self.assertRaises(TypeError, c_char_p.from_param, 42)
    a = c_char_p(b'123')
    self.assertIs(c_char_p.from_param(a), a)
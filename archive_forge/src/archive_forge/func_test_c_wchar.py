import unittest
import sys
from ctypes import *
def test_c_wchar(self):
    x = c_wchar('x')
    self.assertRaises(TypeError, c_wchar, b'x')
    x.value = 'y'
    with self.assertRaises(TypeError):
        x.value = b'y'
    c_wchar.from_param('x')
    self.assertRaises(TypeError, c_wchar.from_param, b'x')
    (c_wchar * 3)('a', 'b', 'c')
    self.assertRaises(TypeError, c_wchar * 3, b'a', b'b', b'c')
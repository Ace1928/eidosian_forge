import unittest
import sys
from ctypes import *
def test_c_char(self):
    x = c_char(b'x')
    self.assertRaises(TypeError, c_char, 'x')
    x.value = b'y'
    with self.assertRaises(TypeError):
        x.value = 'y'
    c_char.from_param(b'x')
    self.assertRaises(TypeError, c_char.from_param, 'x')
    self.assertIn('xbd', repr(c_char.from_param(b'\xbd')))
    (c_char * 3)(b'a', b'b', b'c')
    self.assertRaises(TypeError, c_char * 3, 'a', 'b', 'c')
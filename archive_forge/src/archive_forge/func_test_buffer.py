from ctypes import *
from ctypes.test import need_symbol
import unittest
def test_buffer(self):
    b = create_string_buffer(32)
    self.assertEqual(len(b), 32)
    self.assertEqual(sizeof(b), 32 * sizeof(c_char))
    self.assertIs(type(b[0]), bytes)
    b = create_string_buffer(b'abc')
    self.assertEqual(len(b), 4)
    self.assertEqual(sizeof(b), 4 * sizeof(c_char))
    self.assertIs(type(b[0]), bytes)
    self.assertEqual(b[0], b'a')
    self.assertEqual(b[:], b'abc\x00')
    self.assertEqual(b[:], b'abc\x00')
    self.assertEqual(b[::-1], b'\x00cba')
    self.assertEqual(b[::2], b'ac')
    self.assertEqual(b[::5], b'a')
    self.assertRaises(TypeError, create_string_buffer, 'abc')
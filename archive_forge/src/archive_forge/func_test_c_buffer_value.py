import unittest
from ctypes import *
from ctypes.test import need_symbol
def test_c_buffer_value(self):
    buf = c_buffer(32)
    buf.value = b'Hello, World'
    self.assertEqual(buf.value, b'Hello, World')
    self.assertRaises(TypeError, setattr, buf, 'value', memoryview(b'Hello, World'))
    self.assertRaises(TypeError, setattr, buf, 'value', memoryview(b'abc'))
    self.assertRaises(ValueError, setattr, buf, 'raw', memoryview(b'x' * 100))
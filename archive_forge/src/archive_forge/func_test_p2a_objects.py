from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
def test_p2a_objects(self):
    array = (c_char_p * 5)()
    self.assertEqual(array._objects, None)
    array[0] = b'foo bar'
    self.assertEqual(array._objects, {'0': b'foo bar'})
    p = cast(array, POINTER(c_char_p))
    self.assertIs(p._objects, array._objects)
    self.assertEqual(array._objects, {'0': b'foo bar', id(array): array})
    p[0] = b'spam spam'
    self.assertEqual(p._objects, {'0': b'spam spam', id(array): array})
    self.assertIs(array._objects, p._objects)
    p[1] = b'foo bar'
    self.assertEqual(p._objects, {'1': b'foo bar', '0': b'spam spam', id(array): array})
    self.assertIs(array._objects, p._objects)
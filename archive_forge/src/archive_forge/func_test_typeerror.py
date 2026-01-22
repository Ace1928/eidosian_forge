from ctypes import *
import unittest
import struct
def test_typeerror(self):
    for t in signed_types + unsigned_types + float_types:
        self.assertRaises(TypeError, t, '')
        self.assertRaises(TypeError, t, None)
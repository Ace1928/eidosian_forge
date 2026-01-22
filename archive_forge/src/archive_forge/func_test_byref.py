from ctypes import *
import unittest
import struct
def test_byref(self):
    for t in signed_types + unsigned_types + float_types + bool_types:
        parm = byref(t())
        self.assertEqual(ArgType, type(parm))
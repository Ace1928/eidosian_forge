import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_bad_length(self):
    with self.assertRaises(ValueError):

        class T(Array):
            _type_ = c_int
            _length_ = -sys.maxsize * 2
    with self.assertRaises(ValueError):

        class T2(Array):
            _type_ = c_int
            _length_ = -1
    with self.assertRaises(TypeError):

        class T3(Array):
            _type_ = c_int
            _length_ = 1.87
    with self.assertRaises(OverflowError):

        class T4(Array):
            _type_ = c_int
            _length_ = sys.maxsize * 2
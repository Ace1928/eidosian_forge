import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_bad_subclass(self):
    with self.assertRaises(AttributeError):

        class T(Array):
            pass
    with self.assertRaises(AttributeError):

        class T2(Array):
            _type_ = c_int
    with self.assertRaises(AttributeError):

        class T3(Array):
            _length_ = 13
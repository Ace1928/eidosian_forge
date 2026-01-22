import unittest
from ctypes import *
def test___set__(self):

    class MyCStruct(Structure):
        _fields_ = (('field', c_int),)
    self.assertRaises(TypeError, MyCStruct.field.__set__, 'wrong type self', 42)

    class MyCUnion(Union):
        _fields_ = (('field', c_int),)
    self.assertRaises(TypeError, MyCUnion.field.__set__, 'wrong type self', 42)
import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_union_struct(self):
    for nested, data in ((BigEndianStructure, b'\x00\x00\x00\x01\x00\x00\x00\x02'), (LittleEndianStructure, b'\x01\x00\x00\x00\x02\x00\x00\x00')):
        for parent in (BigEndianUnion, LittleEndianUnion, Union):

            class NestedStructure(nested):
                _fields_ = [('x', c_uint32), ('y', c_uint32)]

            class TestUnion(parent):
                _fields_ = [('point', NestedStructure)]
            self.assertEqual(len(data), sizeof(TestUnion))
            ptr = POINTER(TestUnion)
            s = cast(data, ptr)[0]
            del ctypes._pointer_type_cache[TestUnion]
            self.assertEqual(s.point.x, 1)
            self.assertEqual(s.point.y, 2)
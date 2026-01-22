import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_unaligned_native_struct_fields(self):
    if sys.byteorder == 'little':
        fmt = '<b h xi xd'
    else:
        base = LittleEndianStructure
        fmt = '>b h xi xd'

    class S(Structure):
        _pack_ = 1
        _fields_ = [('b', c_byte), ('h', c_short), ('_1', c_byte), ('i', c_int), ('_2', c_byte), ('d', c_double)]
    s1 = S()
    s1.b = 18
    s1.h = 4660
    s1.i = 305419896
    s1.d = 3.14
    s2 = struct.pack(fmt, 18, 4660, 305419896, 3.14)
    self.assertEqual(bin(s1), bin(s2))
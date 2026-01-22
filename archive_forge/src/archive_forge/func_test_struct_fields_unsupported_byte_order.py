import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_struct_fields_unsupported_byte_order(self):
    fields = [('a', c_ubyte), ('b', c_byte), ('c', c_short), ('d', c_ushort), ('e', c_int), ('f', c_uint), ('g', c_long), ('h', c_ulong), ('i', c_longlong), ('k', c_ulonglong), ('l', c_float), ('m', c_double), ('n', c_char), ('b1', c_byte, 3), ('b2', c_byte, 3), ('b3', c_byte, 2), ('a', c_int * 3 * 3 * 3)]
    for typ in (c_wchar, c_void_p, POINTER(c_int)):
        with self.assertRaises(TypeError):

            class T(BigEndianStructure if sys.byteorder == 'little' else LittleEndianStructure):
                _fields_ = fields + [('x', typ)]
import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_endian_longlong(self):
    if sys.byteorder == 'little':
        self.assertIs(c_longlong.__ctype_le__, c_longlong)
        self.assertIs(c_longlong.__ctype_be__.__ctype_le__, c_longlong)
    else:
        self.assertIs(c_longlong.__ctype_be__, c_longlong)
        self.assertIs(c_longlong.__ctype_le__.__ctype_be__, c_longlong)
    s = c_longlong.__ctype_be__(1311768467294899695)
    self.assertEqual(bin(struct.pack('>q', 1311768467294899695)), '1234567890ABCDEF')
    self.assertEqual(bin(s), '1234567890ABCDEF')
    self.assertEqual(s.value, 1311768467294899695)
    s = c_longlong.__ctype_le__(1311768467294899695)
    self.assertEqual(bin(struct.pack('<q', 1311768467294899695)), 'EFCDAB9078563412')
    self.assertEqual(bin(s), 'EFCDAB9078563412')
    self.assertEqual(s.value, 1311768467294899695)
    s = c_ulonglong.__ctype_be__(1311768467294899695)
    self.assertEqual(bin(struct.pack('>Q', 1311768467294899695)), '1234567890ABCDEF')
    self.assertEqual(bin(s), '1234567890ABCDEF')
    self.assertEqual(s.value, 1311768467294899695)
    s = c_ulonglong.__ctype_le__(1311768467294899695)
    self.assertEqual(bin(struct.pack('<Q', 1311768467294899695)), 'EFCDAB9078563412')
    self.assertEqual(bin(s), 'EFCDAB9078563412')
    self.assertEqual(s.value, 1311768467294899695)
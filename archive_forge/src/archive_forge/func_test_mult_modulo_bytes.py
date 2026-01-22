import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_mult_modulo_bytes(self):
    modmult = self.Integer._mult_modulo_bytes
    res = modmult(4, 5, 19)
    self.assertEqual(res, b'\x01')
    res = modmult(4 - 19, 5, 19)
    self.assertEqual(res, b'\x01')
    res = modmult(4, 5 - 19, 19)
    self.assertEqual(res, b'\x01')
    res = modmult(4 + 19, 5, 19)
    self.assertEqual(res, b'\x01')
    res = modmult(4, 5 + 19, 19)
    self.assertEqual(res, b'\x01')
    modulus = 2 ** 512 - 1
    t1 = 13 ** 100
    t2 = 17 ** 100
    expect = b"\xfa\xb2\x11\x87\xc3(y\x07\xf8\xf1n\xdepq\x0b\xca\xf3\xd3B,\xef\xf2\xfbf\xcc)\x8dZ*\x95\x98r\x96\xa8\xd5\xc3}\xe2q:\xa2'z\xf48\xde%\xef\t\x07\xbc\xc4[C\x8bUE2\x90\xef\x81\xaa:\x08"
    self.assertEqual(expect, modmult(t1, t2, modulus))
    self.assertRaises(ZeroDivisionError, modmult, 4, 5, 0)
    self.assertRaises(ValueError, modmult, 4, 5, -1)
    self.assertRaises(ValueError, modmult, 4, 5, 4)
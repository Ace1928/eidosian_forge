import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_conversion_from_bytes(self):
    Integer = self.Integer
    v1 = Integer.from_bytes(b'\x00')
    self.assertTrue(isinstance(v1, Integer))
    self.assertEqual(0, v1)
    v2 = Integer.from_bytes(b'\x00\x01')
    self.assertEqual(1, v2)
    v3 = Integer.from_bytes(b'\xff\xff')
    self.assertEqual(65535, v3)
    v4 = Integer.from_bytes(b'\x00\x01', 'big')
    self.assertEqual(1, v4)
    v5 = Integer.from_bytes(b'\x00\x01', byteorder='big')
    self.assertEqual(1, v5)
    v6 = Integer.from_bytes(b'\x00\x01', byteorder='little')
    self.assertEqual(256, v6)
    self.assertRaises(ValueError, Integer.from_bytes, b'\t', 'bittle')
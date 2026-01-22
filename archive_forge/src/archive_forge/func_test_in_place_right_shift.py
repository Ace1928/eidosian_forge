import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_in_place_right_shift(self):
    v1, v2, v3 = self.Integers(16, 1, -16)
    v1 >>= 0
    self.assertEqual(v1, 16)
    v1 >>= 1
    self.assertEqual(v1, 8)
    v1 >>= v2
    self.assertEqual(v1, 4)
    v3 >>= 1
    self.assertEqual(v3, -8)

    def l():
        v4 = self.Integer(144)
        v4 >>= -1
    self.assertRaises(ValueError, l)

    def m1():
        v4 = self.Integer(144)
        v4 >>= 2 ** 1000
        return v4
    self.assertEqual(0, m1())

    def m2():
        v4 = self.Integer(-1)
        v4 >>= 2 ** 1000
        return v4
    self.assertEqual(-1, m2())
import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_random_max_bits(self):
    flag = False
    for _ in range(1000):
        a = IntegerNative.random(max_bits=8)
        flag = flag or a < 128
        self.assertFalse(a >= 256)
    self.assertTrue(flag)
    for bits_value in range(1024, 1024 + 8):
        a = IntegerNative.random(max_bits=bits_value)
        self.assertFalse(a >= 2 ** bits_value)
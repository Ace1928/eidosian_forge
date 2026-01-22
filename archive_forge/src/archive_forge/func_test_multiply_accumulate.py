import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_multiply_accumulate(self):
    v1, v2, v3 = self.Integers(4, 3, 2)
    v1.multiply_accumulate(v2, v3)
    self.assertEqual(v1, 10)
    v1.multiply_accumulate(v2, 2)
    self.assertEqual(v1, 16)
    v1.multiply_accumulate(3, v3)
    self.assertEqual(v1, 22)
    v1.multiply_accumulate(1, -2)
    self.assertEqual(v1, 20)
    v1.multiply_accumulate(-2, 1)
    self.assertEqual(v1, 18)
    v1.multiply_accumulate(1, 2 ** 1000)
    self.assertEqual(v1, 18 + 2 ** 1000)
    v1.multiply_accumulate(2 ** 1000, 1)
    self.assertEqual(v1, 18 + 2 ** 1001)
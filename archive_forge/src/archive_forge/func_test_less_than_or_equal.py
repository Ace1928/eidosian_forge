import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_less_than_or_equal(self):
    v1, v2, v3, v4, v5 = self.Integers(13, 13, 14, -4, 2 ** 10)
    self.assertTrue(v1 <= v1)
    self.assertTrue(v1 <= 13)
    self.assertTrue(v1 <= v2)
    self.assertTrue(v1 <= 14)
    self.assertTrue(v1 <= v3)
    self.assertFalse(v1 <= v4)
    self.assertTrue(v1 <= v5)
    self.assertFalse(v5 <= v1)
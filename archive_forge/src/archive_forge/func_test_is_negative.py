import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_is_negative(self):
    v1, v2, v3, v4, v5 = self.Integers(-3 ** 100, -3, 0, 3, 3 ** 100)
    self.assertTrue(v1.is_negative())
    self.assertTrue(v2.is_negative())
    self.assertFalse(v4.is_negative())
    self.assertFalse(v5.is_negative())
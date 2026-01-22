import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_equality_with_ints(self):
    v1, v2, v3 = self.Integers(23, -89, 2 ** 1000)
    self.assertTrue(v1 == 23)
    self.assertTrue(v2 == -89)
    self.assertFalse(v1 == 24)
    self.assertTrue(v3 == 2 ** 1000)
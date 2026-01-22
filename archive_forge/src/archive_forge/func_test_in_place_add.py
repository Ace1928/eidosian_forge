import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_in_place_add(self):
    v1, v2 = self.Integers(10, 20)
    v1 += v2
    self.assertEqual(v1, 30)
    v1 += 10
    self.assertEqual(v1, 40)
    v1 += -1
    self.assertEqual(v1, 39)
    v1 += 2 ** 1000
    self.assertEqual(v1, 39 + 2 ** 1000)
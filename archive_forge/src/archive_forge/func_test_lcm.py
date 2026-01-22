import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_lcm(self):
    v1, v2, v3, v4, v5 = self.Integers(6, 10, 17, -2, 0)
    self.assertTrue(isinstance(v1.lcm(v2), self.Integer))
    self.assertEqual(v1.lcm(v2), 30)
    self.assertEqual(v1.lcm(10), 30)
    self.assertEqual(v1.lcm(v3), 102)
    self.assertEqual(v1.lcm(-2), 6)
    self.assertEqual(v4.lcm(6), 6)
    self.assertEqual(v1.lcm(0), 0)
    self.assertEqual(v5.lcm(0), 0)
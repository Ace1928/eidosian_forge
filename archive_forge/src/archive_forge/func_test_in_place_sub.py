import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_in_place_sub(self):
    v1, v2 = self.Integers(10, 20)
    v1 -= v2
    self.assertEqual(v1, -10)
    v1 -= -100
    self.assertEqual(v1, 90)
    v1 -= 90000
    self.assertEqual(v1, -89910)
    v1 -= -100000
    self.assertEqual(v1, 10090)
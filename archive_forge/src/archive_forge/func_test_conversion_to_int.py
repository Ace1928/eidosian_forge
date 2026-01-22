import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_conversion_to_int(self):
    v1, v2 = self.Integers(-23, 2 ** 1000)
    self.assertEqual(int(v1), -23)
    self.assertEqual(int(v2), 2 ** 1000)
import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_inplace_inverse(self):
    v1, v2 = self.Integers(2, 5)
    v1.inplace_inverse(v2)
    self.assertEqual(v1, 3)
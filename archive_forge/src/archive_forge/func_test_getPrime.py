import math
import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes
def test_getPrime(self):
    """Util.number.getPrime"""
    self.assertRaises(ValueError, number.getPrime, -100)
    self.assertRaises(ValueError, number.getPrime, 0)
    self.assertRaises(ValueError, number.getPrime, 1)
    bits = 4
    for i in range(100):
        x = number.getPrime(bits)
        self.assertEqual(x >= 1 << bits - 1, 1)
        self.assertEqual(x < 1 << bits, 1)
    bits = 512
    x = number.getPrime(bits)
    self.assertNotEqual(x % 2, 0)
    self.assertEqual(x >= 1 << bits - 1, 1)
    self.assertEqual(x < 1 << bits, 1)
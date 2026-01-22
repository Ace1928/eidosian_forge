import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_jacobi_symbol(self):
    data = ((1001, 1, 1), (19, 45, 1), (8, 21, -1), (5, 21, 1), (610, 987, -1), (1001, 9907, -1), (5, 3439601197, -1))
    js = self.Integer.jacobi_symbol
    for k in range(1, 30):
        self.assertEqual(js(k, 1), 1)
    for n in range(1, 30, 2):
        self.assertEqual(js(1, n), 1)
    self.assertRaises(ValueError, js, 6, -2)
    self.assertRaises(ValueError, js, 6, -1)
    self.assertRaises(ValueError, js, 6, 0)
    self.assertRaises(ValueError, js, 0, 0)
    self.assertRaises(ValueError, js, 6, 2)
    self.assertRaises(ValueError, js, 6, 4)
    self.assertRaises(ValueError, js, 6, 6)
    self.assertRaises(ValueError, js, 6, 8)
    for tv in data:
        self.assertEqual(js(tv[0], tv[1]), tv[2])
        self.assertEqual(js(self.Integer(tv[0]), tv[1]), tv[2])
        self.assertEqual(js(tv[0], self.Integer(tv[1])), tv[2])
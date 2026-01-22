import math
import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes
def test_ceil_div(self):
    """Util.number.ceil_div"""
    self.assertRaises(TypeError, number.ceil_div, '1', 1)
    self.assertRaises(ZeroDivisionError, number.ceil_div, 1, 0)
    self.assertRaises(ZeroDivisionError, number.ceil_div, -1, 0)
    self.assertEqual(0, number.ceil_div(0, 1))
    self.assertEqual(1, number.ceil_div(1, 1))
    self.assertEqual(2, number.ceil_div(2, 1))
    self.assertEqual(3, number.ceil_div(3, 1))
    self.assertEqual(0, number.ceil_div(0, 2))
    self.assertEqual(1, number.ceil_div(1, 2))
    self.assertEqual(1, number.ceil_div(2, 2))
    self.assertEqual(2, number.ceil_div(3, 2))
    self.assertEqual(2, number.ceil_div(4, 2))
    self.assertEqual(3, number.ceil_div(5, 2))
    self.assertEqual(0, number.ceil_div(0, 3))
    self.assertEqual(1, number.ceil_div(1, 3))
    self.assertEqual(1, number.ceil_div(2, 3))
    self.assertEqual(1, number.ceil_div(3, 3))
    self.assertEqual(2, number.ceil_div(4, 3))
    self.assertEqual(2, number.ceil_div(5, 3))
    self.assertEqual(2, number.ceil_div(6, 3))
    self.assertEqual(3, number.ceil_div(7, 3))
    self.assertEqual(0, number.ceil_div(0, 4))
    self.assertEqual(1, number.ceil_div(1, 4))
    self.assertEqual(1, number.ceil_div(2, 4))
    self.assertEqual(1, number.ceil_div(3, 4))
    self.assertEqual(1, number.ceil_div(4, 4))
    self.assertEqual(2, number.ceil_div(5, 4))
    self.assertEqual(2, number.ceil_div(6, 4))
    self.assertEqual(2, number.ceil_div(7, 4))
    self.assertEqual(2, number.ceil_div(8, 4))
    self.assertEqual(3, number.ceil_div(9, 4))
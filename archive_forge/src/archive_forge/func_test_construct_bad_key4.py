import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_bad_key4(self):
    y, g, p, q = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q)]
    tup = (y, g, p + 1, q)
    self.assertRaises(ValueError, self.dsa.construct, tup)
    tup = (y, g, p, q + 1)
    self.assertRaises(ValueError, self.dsa.construct, tup)
    tup = (y, 1, p, q)
    self.assertRaises(ValueError, self.dsa.construct, tup)
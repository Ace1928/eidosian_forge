import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_bad_key5(self):
    y, g, p, q, x = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q, self.x)]
    tup = (y, g, p, q, x + 1)
    self.assertRaises(ValueError, self.dsa.construct, tup)
    tup = (y, g, p, q, q + 10)
    self.assertRaises(ValueError, self.dsa.construct, tup)
import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_5tuple(self):
    """DSA (default implementation) constructed key (5-tuple)"""
    y, g, p, q, x = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q, self.x)]
    dsaObj = self.dsa.construct((y, g, p, q, x))
    self._test_signing(dsaObj)
    self._test_verification(dsaObj)
import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_error_weak_domain(self):
    """Verify that domain parameters with composite q are rejected"""
    from Cryptodome.Math.Numbers import Integer
    p, q, g = self._get_weak_domain()
    y = pow(g, 89, p)
    self.assertRaises(ValueError, DSA.construct, (y, g, p, q))
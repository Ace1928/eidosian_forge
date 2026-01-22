import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_domain1(self):
    """Verify we can generate new keys in a given domain"""
    dsa_key_1 = DSA.generate(1024)
    domain_params = dsa_key_1.domain()
    dsa_key_2 = DSA.generate(1024, domain=domain_params)
    self.assertEqual(dsa_key_1.p, dsa_key_2.p)
    self.assertEqual(dsa_key_1.q, dsa_key_2.q)
    self.assertEqual(dsa_key_1.g, dsa_key_2.g)
    self.assertEqual(dsa_key_1.domain(), dsa_key_2.domain())
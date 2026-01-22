import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_generate_2arg(self):
    """DSA (default implementation) generated key (2 arguments)"""
    dsaObj = self.dsa.generate(1024, Random.new().read)
    self._check_private_key(dsaObj)
    pub = dsaObj.public_key()
    self._check_public_key(pub)
import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_generate_3args(self):
    rsaObj = self.rsa.generate(1024, Random.new().read, e=65537)
    self._check_private_key(rsaObj)
    self._exercise_primitive(rsaObj)
    pub = rsaObj.public_key()
    self._check_public_key(pub)
    self._exercise_public_primitive(rsaObj)
    self.assertEqual(65537, rsaObj.e)
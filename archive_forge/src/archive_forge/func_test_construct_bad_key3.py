import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_bad_key3(self):
    tup = (self.n, self.e, self.d + 1)
    self.assertRaises(ValueError, self.rsa.construct, tup)
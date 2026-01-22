import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
from Cryptodome import Random
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.py3compat import *
def test_bad_key4(self):
    tup = tup0 = list(self.convert_tv(self.tvs[0], 1)['key'])
    tup[3] += 1
    self.assertRaises(ValueError, ElGamal.construct, tup)
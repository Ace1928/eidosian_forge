import unittest
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import cSHAKE128, cSHAKE256, SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, tobytes
def test_left_encode(self):
    from Cryptodome.Hash.cSHAKE128 import _left_encode
    self.assertEqual(_left_encode(0), b'\x01\x00')
    self.assertEqual(_left_encode(1), b'\x01\x01')
    self.assertEqual(_left_encode(256), b'\x02\x01\x00')
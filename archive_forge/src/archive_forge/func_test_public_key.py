import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def test_public_key(self):
    p521 = _curves['p521']
    point = EccPoint(p521.Gx, p521.Gy, 'p521')
    key = EccKey(curve='P-384', point=point)
    self.assertFalse(key.has_private())
    self.assertEqual(key.pointQ, point)
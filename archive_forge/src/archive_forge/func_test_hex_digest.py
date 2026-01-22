import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
def test_hex_digest(self):
    mac = self.new()
    digest = mac.digest()
    hexdigest = mac.hexdigest()
    self.assertEqual(hexlify(digest), tobytes(hexdigest))
    self.assertEqual(mac.hexdigest(), hexdigest)
    self.assertTrue(isinstance(hexdigest, type('digest')))
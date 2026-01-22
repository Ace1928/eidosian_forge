import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
def test_default_digest_size(self):
    digest = self.new().digest()
    self.assertEqual(len(digest), self.default_bytes)
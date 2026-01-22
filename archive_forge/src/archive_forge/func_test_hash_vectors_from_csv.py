import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_vectors_from_csv(self):
    """Test hash function with precalculated combinations."""
    for row in self.hashes[1:]:
        h = scrypt.hash(row[0], row[1], int(row[2]), int(row[3]), int(row[4]))
        hhex = b2a_hex(h)
        self.assertEqual(hhex, bytes(row[5].encode('utf-8')))
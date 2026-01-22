import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_raises_error_r_p_over_limit(self):
    """Test hash raises scrypt error when parameters r multiplied by p over limit
        2**30."""
    self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, r=2, p=2 ** 29))
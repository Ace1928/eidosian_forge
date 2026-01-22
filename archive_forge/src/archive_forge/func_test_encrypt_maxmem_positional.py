import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_maxmem_positional(self):
    """Test encrypt maxmem accepts 4th positional argument and exactly (1 megabyte)
        of storage to use for V array."""
    s = scrypt.encrypt(self.input, self.password, 0.01, self.one_megabyte)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)
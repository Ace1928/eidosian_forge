import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_maxtime_positional(self):
    """Test encrypt maxtime accepts maxtime at position 3."""
    s = scrypt.encrypt(self.input, self.password, 0.01)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)
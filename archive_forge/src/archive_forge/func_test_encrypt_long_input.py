import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_long_input(self):
    """Test encrypt accepts long input for encryption."""
    s = scrypt.encrypt(self.longinput, self.password, 0.1)
    self.assertEqual(len(s), 128 + len(self.longinput))
import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_decrypt(self):
    """Test encrypt for simple encryption and decryption."""
    s = scrypt.encrypt(self.input, self.password, 0.1)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)
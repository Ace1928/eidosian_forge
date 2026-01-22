import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_missing_password_positional_argument(self):
    """Test encrypt raises TypeError if second positional argument missing
        (password)"""
    self.assertRaises(TypeError, lambda: scrypt.encrypt(self.input))
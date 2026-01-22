import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_raises_error_on_invalid_keyword(self):
    """Test encrypt raises TypeError if invalid keyword used in argument."""
    self.assertRaises(TypeError, lambda: scrypt.encrypt(self.input, self.password, nonsense='Raise error'))
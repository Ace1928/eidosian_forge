import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_input_and_password_as_keywords(self):
    """Test encrypt for input and password accepted as keywords."""
    s = scrypt.encrypt(password=self.password, input=self.input)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)
import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_decrypt_raises_error_on_too_little_time(self):
    """Test decrypt function raises scrypt.error raised if insufficient time allowed
        for ciphertext decryption."""
    s = scrypt.encrypt(self.input, self.password, 0.1)
    self.assertRaises(scrypt.error, lambda: scrypt.decrypt(s, self.password, 0.01))
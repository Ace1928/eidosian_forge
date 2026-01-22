import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_unaligned_data_128(self):
    cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
    for wrong_length in range(1, 16):
        self.assertRaises(ValueError, cipher.encrypt, b'5' * wrong_length)
    cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
    for wrong_length in range(1, 16):
        self.assertRaises(ValueError, cipher.decrypt, b'5' * wrong_length)
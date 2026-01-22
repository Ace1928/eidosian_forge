import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_only_one_iv(self):
    self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode, iv=self.iv_128, IV=self.iv_128)
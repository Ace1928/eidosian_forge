import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_IV_iv_attributes(self):
    data = get_tag_random('data', 16 * 100)
    for func in ('encrypt', 'decrypt'):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        getattr(cipher, func)(data)
        self.assertEqual(cipher.iv, self.iv_128)
        self.assertEqual(cipher.IV, self.iv_128)
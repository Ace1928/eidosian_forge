import json
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_invalid_multiple_decrypt_and_verify(self):
    cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
    ct, tag = cipher.encrypt_and_digest(self.data)
    cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
    cipher.decrypt_and_verify(ct, tag)
    self.assertRaises(TypeError, cipher.decrypt_and_verify, ct, tag)
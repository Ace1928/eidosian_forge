import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_shorter_ciphertext_than_declared(self):
    DATA_LEN = len(self.data)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
    ct, mac = cipher.encrypt_and_digest(self.data)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, msg_len=DATA_LEN + 1)
    cipher.decrypt(ct)
    self.assertRaises(ValueError, cipher.verify, mac)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, msg_len=DATA_LEN - 1)
    self.assertRaises(ValueError, cipher.decrypt, ct)
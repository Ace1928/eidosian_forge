import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_shorter_assoc_data_than_expected(self):
    DATA_LEN = len(self.data)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, assoc_len=DATA_LEN + 1)
    cipher.update(self.data)
    self.assertRaises(ValueError, cipher.encrypt, self.data)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, assoc_len=DATA_LEN + 1)
    cipher.update(self.data)
    self.assertRaises(ValueError, cipher.digest)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, assoc_len=DATA_LEN + 1)
    cipher.update(self.data)
    self.assertRaises(ValueError, cipher.decrypt, self.data)
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
    cipher.update(self.data)
    mac = cipher.digest()
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, assoc_len=DATA_LEN + 1)
    cipher.update(self.data)
    self.assertRaises(ValueError, cipher.verify, mac)
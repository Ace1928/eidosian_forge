import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
def test_initial_value_bytes_parameter(self):
    cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=b'\x00' * 6 + b'\xff\xff')
    cipher2 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=65535)
    pt = get_tag_random('plaintext', 65536)
    self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
    self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, initial_value=b'5' * 17)
    self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=b'5' * 9)
    self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, initial_value=b'5' * 15)
    self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=b'5' * 7)
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_invalid_decrypt_or_update_after_verify(self):
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    ct = cipher.encrypt(self.data)
    ct += cipher.encrypt()
    mac = cipher.digest()
    for method_name in ('decrypt', 'update'):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.decrypt(ct)
        cipher.decrypt()
        cipher.verify(mac)
        self.assertRaises(TypeError, getattr(cipher, method_name), self.data)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.decrypt_and_verify(ct, mac)
        self.assertRaises(TypeError, getattr(cipher, method_name), self.data)
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_invalid_mixing_encrypt_decrypt(self):
    for method1_name, method2_name in (('encrypt', 'decrypt'), ('decrypt', 'encrypt')):
        for assoc_data_present in (True, False):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            if assoc_data_present:
                cipher.update(self.data)
            getattr(cipher, method1_name)(self.data)
            self.assertRaises(TypeError, getattr(cipher, method2_name), self.data)
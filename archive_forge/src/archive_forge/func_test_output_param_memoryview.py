import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
def test_output_param_memoryview(self):
    pt = b'5' * 128
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    ct = cipher.encrypt(pt)
    output = memoryview(bytearray(128))
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    cipher.encrypt(pt, output=output)
    self.assertEqual(ct, output)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    cipher.decrypt(ct, output=output)
    self.assertEqual(pt, output)
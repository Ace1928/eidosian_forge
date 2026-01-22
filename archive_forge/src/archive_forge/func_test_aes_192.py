import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_aes_192(self):
    key = '8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b'
    iv = '000102030405060708090a0b0c0d0e0f'
    plaintext = '6bc1bee22e409f96e93d7e117393172a' + 'ae2d8a571e03ac9c9eb76fac45af8e51' + '30c81c46a35ce411e5fbc1191a0a52ef' + 'f69f2445df4f9b17ad2b417be66c3710'
    ciphertext = '4f021db243bc633d7178183a9fa071e8' + 'b4d9ada9ad7dedf4e5e738763f69145a' + '571b242012fb7ae07fa9baac3df102e0' + '08b0e27988598881d920a9e64f5615cd'
    key = unhexlify(key)
    iv = unhexlify(iv)
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    self.assertEqual(cipher.encrypt(plaintext), ciphertext)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    self.assertEqual(cipher.decrypt(ciphertext), plaintext)
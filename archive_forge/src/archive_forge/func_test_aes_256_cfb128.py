import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_aes_256_cfb128(self):
    plaintext = '6bc1bee22e409f96e93d7e117393172a' + 'ae2d8a571e03ac9c9eb76fac45af8e51' + '30c81c46a35ce411e5fbc1191a0a52ef' + 'f69f2445df4f9b17ad2b417be66c3710'
    ciphertext = 'dc7e84bfda79164b7ecd8486985d3860' + '39ffed143b28b1c832113c6331e5407b' + 'df10132415e54b92a13ed0a8267ae2f9' + '75a385741ab9cef82031623d55b1e471'
    key = '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4'
    iv = '000102030405060708090a0b0c0d0e0f'
    key = unhexlify(key)
    iv = unhexlify(iv)
    plaintext = unhexlify(plaintext)
    ciphertext = unhexlify(ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    self.assertEqual(cipher.encrypt(plaintext), ciphertext)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    self.assertEqual(cipher.decrypt(ciphertext), plaintext)
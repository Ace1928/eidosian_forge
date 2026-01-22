import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.IO import PKCS8
from Cryptodome.Util.asn1 import DerNull
def test_import_botan_keys(self):
    botan_scrypt_der = txt2bin(botan_scrypt)
    key1 = PKCS8.unwrap(botan_scrypt_der, b'your_password')
    botan_pbkdf2_der = txt2bin(botan_pbkdf2)
    key2 = PKCS8.unwrap(botan_pbkdf2_der, b'your_password')
    self.assertEqual(key1, key2)
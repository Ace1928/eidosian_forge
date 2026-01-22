import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
def test_export_private_pkcs8_and_pem_2(self):
    encoded = self.ref_private._export_private_encrypted_pkcs8_in_clear_pem('secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
    self.assertRaises(ValueError, ECC.import_key, encoded)
    assert 'ENCRYPTED PRIVATE KEY' in encoded
    decoded = ECC.import_key(encoded, 'secret')
    self.assertEqual(self.ref_private, decoded)
    encoded = self.ref_private.export_key(format='PEM', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
    decoded = ECC.import_key(encoded, 'secret')
    self.assertEqual(self.ref_private, decoded)
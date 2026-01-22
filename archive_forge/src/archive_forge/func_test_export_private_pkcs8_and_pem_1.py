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
def test_export_private_pkcs8_and_pem_1(self):
    key_file = load_file('ecc_p521_private_p8_clear.pem', 'rt').strip()
    encoded = self.ref_private._export_private_clear_pkcs8_in_clear_pem()
    self.assertEqual(key_file, encoded)
    encoded = self.ref_private.export_key(format='PEM')
    self.assertEqual(key_file, encoded)
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
def test_export_public_pem_compressed(self):
    key_file = load_file('ecc_p521_public.pem', 'rt').strip()
    pub_key = ECC.import_key(key_file)
    key_file_compressed = pub_key.export_key(format='PEM', compress=True)
    key_file_compressed_ref = load_file('ecc_p521_public_compressed.pem', 'rt').strip()
    self.assertEqual(key_file_compressed, key_file_compressed_ref)
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
def test_export_public_pem(self):
    key_file_ref = load_file('ecc_ed448_public.pem', 'rt').strip()
    key_file = self.ref_public.export_key(format='PEM').strip()
    self.assertEqual(key_file_ref, key_file)
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
def test_export_raw(self):
    encoded = self.ref_public.export_key(format='raw')
    self.assertEqual(encoded, unhexlify(b'899014ddc0a0e1260cfc1085afdf952019e9fd63372e3e366e26dad32b176624884330a14617237e3081febd9d1a15069e7499433d2f55dd80'))
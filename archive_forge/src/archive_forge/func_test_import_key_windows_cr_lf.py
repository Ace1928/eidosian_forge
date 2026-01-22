import os
import re
import errno
import warnings
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import a2b_hex, list_test_cases
from Cryptodome.IO import PEM
from Cryptodome.Util.py3compat import b, tostr, FileNotFoundError
from Cryptodome.Util.number import inverse, bytes_to_long
from Cryptodome.Util import asn1
def test_import_key_windows_cr_lf(self):
    pem_cr_lf = '\r\n'.join(self.rsaKeyPEM.splitlines())
    key = RSA.importKey(pem_cr_lf)
    self.assertEqual(key.n, self.n)
    self.assertEqual(key.e, self.e)
    self.assertEqual(key.d, self.d)
    self.assertEqual(key.p, self.p)
    self.assertEqual(key.q, self.q)
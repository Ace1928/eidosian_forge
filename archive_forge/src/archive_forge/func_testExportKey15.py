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
def testExportKey15(self):
    key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
    self.assertRaises(ValueError, key.export_key, 'DER', 'test', 1)
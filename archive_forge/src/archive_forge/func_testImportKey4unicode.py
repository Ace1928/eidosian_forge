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
def testImportKey4unicode(self):
    """Verify import of RSAPrivateKey DER SEQUENCE, encoded with PEM as unicode"""
    key = RSA.importKey(self.rsaPublicKeyPEM)
    self.assertEqual(key.has_private(), False)
    self.assertEqual(key.n, self.n)
    self.assertEqual(key.e, self.e)
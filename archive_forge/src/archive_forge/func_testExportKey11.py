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
def testExportKey11(self):
    key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
    outkey = key.export_key('PEM', 'test', pkcs=1)
    self.assertTrue(tostr(outkey).find('4,ENCRYPTED') != -1)
    self.assertTrue(tostr(outkey).find('BEGIN RSA PRIVATE KEY') != -1)
    inkey = RSA.importKey(outkey, 'test')
    self.assertEqual(key.n, inkey.n)
    self.assertEqual(key.e, inkey.e)
    self.assertEqual(key.d, inkey.d)
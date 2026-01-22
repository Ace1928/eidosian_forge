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
def testExportKey16(self):
    key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
    outkey = key.export_key('PEM', 'test', pkcs=8, protection='PBKDF2WithHMAC-SHA512AndAES256-CBC', prot_params={'iteration_count': 123})
    self.assertTrue(tostr(outkey).find('4,ENCRYPTED') == -1)
    self.assertTrue(tostr(outkey).find('BEGIN ENCRYPTED PRIVATE KEY') != -1)
    der = PEM.decode(tostr(outkey))[0]
    seq1 = asn1.DerSequence().decode(der)
    seq2 = asn1.DerSequence().decode(seq1[0])
    seq3 = asn1.DerSequence().decode(seq2[1])
    seq4 = asn1.DerSequence().decode(seq3[0])
    seq5 = asn1.DerSequence().decode(seq4[1])
    self.assertEqual(seq5[1], 123)
    inkey = RSA.importKey(outkey, 'test')
    self.assertEqual(key.n, inkey.n)
    self.assertEqual(key.e, inkey.e)
    self.assertEqual(key.d, inkey.d)
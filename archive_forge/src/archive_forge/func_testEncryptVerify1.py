from __future__ import print_function
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome import Random
from Cryptodome.Cipher import PKCS1_v1_5 as PKCS
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def testEncryptVerify1(self):
    for pt_len in range(0, 128 - 11 + 1):
        pt = self.rng(pt_len)
        cipher = PKCS.new(self.key1024)
        ct = cipher.encrypt(pt)
        pt2 = cipher.decrypt(ct, b'\xaa' * pt_len)
        self.assertEqual(pt, pt2)
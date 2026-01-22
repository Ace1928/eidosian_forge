import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP as PKCS
from Cryptodome.Hash import MD2, MD5, SHA1, SHA256, RIPEMD160, SHA224, SHA384, SHA512
from Cryptodome import Random
from Cryptodome.Signature.pss import MGF1
from Cryptodome.Util.py3compat import b, bchr
def testDecrypt1(self):
    for test in self._testData:
        comps = [int(rws(test[0][x]), 16) for x in ('n', 'e', 'd')]
        key = RSA.construct(comps)
        cipher = PKCS.new(key, test[4])
        pt = cipher.decrypt(t2b(test[2]))
        self.assertEqual(pt, t2b(test[1]))
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP as PKCS
from Cryptodome.Hash import MD2, MD5, SHA1, SHA256, RIPEMD160, SHA224, SHA384, SHA512
from Cryptodome import Random
from Cryptodome.Signature.pss import MGF1
from Cryptodome.Util.py3compat import b, bchr
def testEncryptDecrypt3(self):
    pt = self.rng(35)
    xlabel = self.rng(22)
    cipher = PKCS.new(self.key1024, label=xlabel)
    ct = cipher.encrypt(pt)
    self.assertEqual(cipher.decrypt(ct), pt)
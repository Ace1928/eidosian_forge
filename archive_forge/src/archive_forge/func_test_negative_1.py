import unittest
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, SHA224, SHA256, SHA384, SHA512
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pss
from Cryptodome.Signature import PKCS1_PSS
from Cryptodome.Signature.pss import MGF1
def test_negative_1(self):
    key = RSA.import_key(self.rsa_key)
    h = SHA256.new(self.msg + b'A')
    verifier = pss.new(key)
    tag = bytearray(self.tag)
    self.assertRaises(ValueError, verifier.verify, h, tag)
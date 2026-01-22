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
def verify_positive(self, hashmod, message, public_key, salt, signature):
    prng = PRNG(salt)
    hashed = hashmod.new(message)
    verifier = pss.new(public_key, salt_bytes=len(salt), rand_func=prng)
    verifier.verify(hashed, signature)
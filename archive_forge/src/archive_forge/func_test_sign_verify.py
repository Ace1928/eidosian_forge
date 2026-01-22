import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
def test_sign_verify(self):
    """Verify public/private method"""
    self.description = 'can_sign() test'
    signer = DSS.new(self.key_priv, 'fips-186-3')
    self.assertTrue(signer.can_sign())
    signer = DSS.new(self.key_pub, 'fips-186-3')
    self.assertFalse(signer.can_sign())
    self.assertRaises(TypeError, signer.sign, SHA256.new(b'xyz'))
    try:
        signer.sign(SHA256.new(b'xyz'))
    except TypeError as e:
        msg = str(e)
    else:
        msg = ''
    self.assertTrue('Private key is needed' in msg)
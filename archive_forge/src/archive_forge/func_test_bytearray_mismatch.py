import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
def test_bytearray_mismatch(self):
    ref = bcrypt('pwd', 4)
    bcrypt_check('pwd', ref)
    bref = bytearray(ref)
    bcrypt_check('pwd', bref)
    wrong = ref[:-1] + bchr(bref[-1] ^ 1)
    self.assertRaises(ValueError, bcrypt_check, 'pwd', wrong)
    wrong = b'x' + ref[1:]
    self.assertRaises(ValueError, bcrypt_check, 'pwd', wrong)
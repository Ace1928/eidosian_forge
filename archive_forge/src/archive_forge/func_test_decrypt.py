from __future__ import print_function
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome import Random
from Cryptodome.Cipher import PKCS1_v1_5 as PKCS
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def test_decrypt(self, tv):
    self._id = 'Wycheproof Decrypt PKCS#1v1.5 Test #%s' % tv.id
    sentinel = b'\xaa' * max(3, len(tv.msg))
    cipher = PKCS.new(tv.rsa_key)
    try:
        pt = cipher.decrypt(tv.ct, sentinel=sentinel)
    except ValueError:
        assert not tv.valid
    else:
        if pt == sentinel:
            assert not tv.valid
        else:
            assert tv.valid
            self.assertEqual(pt, tv.msg)
            self.warn(tv)
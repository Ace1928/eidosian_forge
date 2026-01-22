import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
def test_oid(self):
    prefix = '1.3.6.1.4.1.1722.12.2.' + self.oid_variant + '.'
    for digest_bits in self.digest_bits_oid:
        h = self.BLAKE2.new(digest_bits=digest_bits)
        self.assertEqual(h.oid, prefix + str(digest_bits // 8))
        h = self.BLAKE2.new(digest_bits=digest_bits, key=b'secret')
        self.assertRaises(AttributeError, lambda: h.oid)
    for digest_bits in (8, self.max_bits):
        if digest_bits in self.digest_bits_oid:
            continue
        self.assertRaises(AttributeError, lambda: h.oid)
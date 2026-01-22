from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
def test_md4_digest(self):
    """digest() method"""
    md4 = self.get_md4_const()
    for input, hex in self.vectors:
        out = bascii_to_str(hexlify(md4(input).digest()))
        self.assertEqual(out, hex)
from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
def test_md4_hexdigest(self):
    """hexdigest() method"""
    md4 = self.get_md4_const()
    for input, hex in self.vectors:
        out = md4(input).hexdigest()
        self.assertEqual(out, hex)
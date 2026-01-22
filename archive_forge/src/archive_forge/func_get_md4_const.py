from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
def get_md4_const(self):
    """
        get md4 constructor --
        overridden by subclasses to use alternate backends.
        """
    return lookup_hash('md4').const
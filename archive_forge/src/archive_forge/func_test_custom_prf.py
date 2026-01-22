from __future__ import with_statement
import hashlib
import warnings
from passlib.utils.compat import u, JYTHON
from passlib.tests.utils import TestCase, hb
def test_custom_prf(self):
    """test custom prf function"""
    from passlib.utils.pbkdf2 import pbkdf2

    def prf(key, msg):
        return hashlib.md5(key + msg + b'fooey').digest()
    self.assertRaises(NotImplementedError, pbkdf2, b'secret', b'salt', 1000, 20, prf)
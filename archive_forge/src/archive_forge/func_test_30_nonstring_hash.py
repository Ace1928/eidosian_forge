from __future__ import with_statement
from logging import getLogger
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, CryptPolicy, LazyCryptContext
from passlib.utils import to_bytes, to_unicode
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase, set_file
from passlib.registry import (register_crypt_handler_path,
def test_30_nonstring_hash(self):
    """test non-string hash values cause error"""
    warnings.filterwarnings('ignore', ".*needs_update.*'scheme' keyword is deprecated.*")
    cc = CryptContext(['des_crypt'])
    for hash, kwds in [(None, {}), (None, {'scheme': 'des_crypt'}), (1, {}), ((), {})]:
        self.assertRaises(TypeError, cc.hash_needs_update, hash, **kwds)
    cc2 = CryptContext(['mysql323'])
    self.assertRaises(TypeError, cc2.hash_needs_update, None)
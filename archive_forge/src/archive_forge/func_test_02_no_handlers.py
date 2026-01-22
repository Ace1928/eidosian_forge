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
def test_02_no_handlers(self):
    """test no handlers"""
    cc = CryptContext()
    self.assertRaises(KeyError, cc.identify, 'hash', required=True)
    self.assertRaises(KeyError, cc.hash, 'secret')
    self.assertRaises(KeyError, cc.verify, 'secret', 'hash')
    cc = CryptContext(['md5_crypt'])
    p = CryptPolicy(schemes=[])
    cc.policy = p
    self.assertRaises(KeyError, cc.identify, 'hash', required=True)
    self.assertRaises(KeyError, cc.hash, 'secret')
    self.assertRaises(KeyError, cc.verify, 'secret', 'hash')
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
def test_12_get_handler(self):
    """test get_handler() method"""
    p1 = CryptPolicy(**self.sample_config_1pd)
    self.assertIs(p1.get_handler('bsdi_crypt'), hash.bsdi_crypt)
    self.assertIs(p1.get_handler('sha256_crypt'), None)
    self.assertRaises(KeyError, p1.get_handler, 'sha256_crypt', required=True)
    self.assertIs(p1.get_handler(), hash.md5_crypt)
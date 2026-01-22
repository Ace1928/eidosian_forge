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
def test_13_get_options(self):
    """test get_options() method"""
    p12 = CryptPolicy(**self.sample_config_12pd)
    self.assertEqual(p12.get_options('bsdi_crypt'), dict(vary_rounds=0.1, min_rounds=29000, max_rounds=35000, default_rounds=31000))
    self.assertEqual(p12.get_options('sha512_crypt'), dict(vary_rounds=0.1, min_rounds=45000, max_rounds=50000))
    p4 = CryptPolicy.from_string(self.sample_config_4s)
    self.assertEqual(p4.get_options('sha512_crypt'), dict(vary_rounds=0.1, max_rounds=20000))
    self.assertEqual(p4.get_options('sha512_crypt', 'user'), dict(vary_rounds=0.1, max_rounds=20000))
    self.assertEqual(p4.get_options('sha512_crypt', 'admin'), dict(vary_rounds=0.05, max_rounds=40000))
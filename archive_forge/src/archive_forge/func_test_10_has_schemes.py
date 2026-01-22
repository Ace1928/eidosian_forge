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
def test_10_has_schemes(self):
    """test has_schemes() method"""
    p1 = CryptPolicy(**self.sample_config_1pd)
    self.assertTrue(p1.has_schemes())
    p3 = CryptPolicy(**self.sample_config_3pd)
    self.assertTrue(not p3.has_schemes())
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
def test_22_to_string(self):
    """test to_string() method"""
    pa = CryptPolicy(**self.sample_config_5pd)
    s = pa.to_string()
    pb = CryptPolicy.from_string(s)
    self.assertEqual(pb.to_dict(), self.sample_config_5pd)
    s = pa.to_string(encoding='latin-1')
    self.assertIsInstance(s, bytes)
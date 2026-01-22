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
def test_03_from_source(self):
    """test CryptPolicy.from_source() constructor"""
    policy = CryptPolicy.from_source(self.sample_config_1s_path)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    policy = CryptPolicy.from_source(self.sample_config_1s)
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    policy = CryptPolicy.from_source(self.sample_config_1pd.copy())
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    p2 = CryptPolicy.from_source(policy)
    self.assertIs(policy, p2)
    self.assertRaises(TypeError, CryptPolicy.from_source, 1)
    self.assertRaises(TypeError, CryptPolicy.from_source, [])
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
def test_04_from_sources(self):
    """test CryptPolicy.from_sources() constructor"""
    self.assertRaises(ValueError, CryptPolicy.from_sources, [])
    policy = CryptPolicy.from_sources([self.sample_config_1s])
    self.assertEqual(policy.to_dict(), self.sample_config_1pd)
    policy = CryptPolicy.from_sources([self.sample_config_1s_path, self.sample_config_2s, self.sample_config_3pd])
    self.assertEqual(policy.to_dict(), self.sample_config_123pd)
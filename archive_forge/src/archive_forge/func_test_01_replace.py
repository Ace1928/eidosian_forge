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
def test_01_replace(self):
    """test replace()"""
    cc = CryptContext(['md5_crypt', 'bsdi_crypt', 'des_crypt'])
    self.assertIs(cc.policy.get_handler(), hash.md5_crypt)
    cc2 = cc.replace()
    self.assertIsNot(cc2, cc)
    cc3 = cc.replace(default='bsdi_crypt')
    self.assertIsNot(cc3, cc)
    self.assertIs(cc3.policy.get_handler(), hash.bsdi_crypt)
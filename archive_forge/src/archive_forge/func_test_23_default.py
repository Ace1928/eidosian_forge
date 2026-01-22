from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
def test_23_default(self):
    """test 'default' context option parsing"""
    self.assertEqual(CryptContext(default='md5_crypt').to_dict(), dict(default='md5_crypt'))
    ctx = CryptContext(default='md5_crypt', schemes=['des_crypt', 'md5_crypt'])
    self.assertEqual(ctx.default_scheme(), 'md5_crypt')
    ctx = CryptContext(default=hash.md5_crypt, schemes=['des_crypt', 'md5_crypt'])
    self.assertEqual(ctx.default_scheme(), 'md5_crypt')
    ctx = CryptContext(schemes=['des_crypt', 'md5_crypt'])
    self.assertEqual(ctx.default_scheme(), 'des_crypt')
    ctx.update(deprecated='des_crypt')
    self.assertEqual(ctx.default_scheme(), 'md5_crypt')
    self.assertRaises(KeyError, CryptContext, schemes=['des_crypt'], default='md5_crypt')
    self.assertRaises(TypeError, CryptContext, default=1)
    ctx = CryptContext(default='des_crypt', schemes=['des_crypt', 'md5_crypt'], admin__context__default='md5_crypt')
    self.assertEqual(ctx.default_scheme(), 'des_crypt')
    self.assertEqual(ctx.default_scheme('user'), 'des_crypt')
    self.assertEqual(ctx.default_scheme('admin'), 'md5_crypt')
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
def test_31_default_scheme(self):
    """test default_scheme() method"""
    ctx = CryptContext()
    self.assertRaises(KeyError, ctx.default_scheme)
    ctx = CryptContext(**self.sample_1_dict)
    self.assertEqual(ctx.default_scheme(), 'md5_crypt')
    self.assertEqual(ctx.default_scheme(resolve=True, unconfigured=True), hash.md5_crypt)
    self.assertHandlerDerivedFrom(ctx.default_scheme(resolve=True), hash.md5_crypt)
    ctx = CryptContext(**self.sample_2_dict)
    self.assertRaises(KeyError, ctx.default_scheme)
    ctx = CryptContext(schemes=self.sample_1_schemes)
    self.assertEqual(ctx.default_scheme(), 'des_crypt')
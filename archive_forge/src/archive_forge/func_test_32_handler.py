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
def test_32_handler(self):
    """test handler() method"""
    ctx = CryptContext()
    self.assertRaises(KeyError, ctx.handler)
    self.assertRaises(KeyError, ctx.handler, 'md5_crypt')
    ctx = CryptContext(**self.sample_1_dict)
    self.assertEqual(ctx.handler(unconfigured=True), hash.md5_crypt)
    self.assertHandlerDerivedFrom(ctx.handler(), hash.md5_crypt)
    self.assertEqual(ctx.handler('des_crypt', unconfigured=True), hash.des_crypt)
    self.assertHandlerDerivedFrom(ctx.handler('des_crypt'), hash.des_crypt)
    self.assertRaises(KeyError, ctx.handler, 'mysql323')
    ctx = CryptContext('sha256_crypt,md5_crypt', admin__context__default='md5_crypt')
    self.assertEqual(ctx.handler(unconfigured=True), hash.sha256_crypt)
    self.assertHandlerDerivedFrom(ctx.handler(), hash.sha256_crypt)
    self.assertEqual(ctx.handler(category='staff', unconfigured=True), hash.sha256_crypt)
    self.assertHandlerDerivedFrom(ctx.handler(category='staff'), hash.sha256_crypt)
    self.assertEqual(ctx.handler(category='admin', unconfigured=True), hash.md5_crypt)
    self.assertHandlerDerivedFrom(ctx.handler(category='staff'), hash.sha256_crypt)
    if PY2:
        self.assertEqual(ctx.handler(category=u('staff'), unconfigured=True), hash.sha256_crypt)
        self.assertEqual(ctx.handler(category=u('admin'), unconfigured=True), hash.md5_crypt)
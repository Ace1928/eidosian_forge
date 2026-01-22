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
def test_30_schemes(self):
    """test schemes() method"""
    ctx = CryptContext()
    self.assertEqual(ctx.schemes(), ())
    self.assertEqual(ctx.schemes(resolve=True), ())
    ctx = CryptContext(**self.sample_1_dict)
    self.assertEqual(ctx.schemes(), tuple(self.sample_1_schemes))
    self.assertEqual(ctx.schemes(resolve=True, unconfigured=True), tuple(self.sample_1_handlers))
    for result, correct in zip(ctx.schemes(resolve=True), self.sample_1_handlers):
        self.assertTrue(handler_derived_from(result, correct))
    ctx = CryptContext(**self.sample_2_dict)
    self.assertEqual(ctx.schemes(), ())
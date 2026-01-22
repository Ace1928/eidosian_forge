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
def test_01_constructor(self):
    """test class constructor"""
    ctx = CryptContext()
    self.assertEqual(ctx.to_dict(), {})
    ctx = CryptContext(**self.sample_1_dict)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx = CryptContext(**self.sample_1_resolved_dict)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx = CryptContext(**self.sample_2_dict)
    self.assertEqual(ctx.to_dict(), self.sample_2_dict)
    ctx = CryptContext(**self.sample_3_dict)
    self.assertEqual(ctx.to_dict(), self.sample_3_dict)
    ctx = CryptContext(schemes=[u('sha256_crypt')])
    self.assertEqual(ctx.schemes(), ('sha256_crypt',))
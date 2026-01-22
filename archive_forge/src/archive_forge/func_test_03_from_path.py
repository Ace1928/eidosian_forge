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
def test_03_from_path(self):
    """test from_path() constructor"""
    if not os.path.exists(self.sample_1_path):
        raise RuntimeError("can't find data file: %r" % self.sample_1_path)
    ctx = CryptContext.from_path(self.sample_1_path)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx = CryptContext.from_path(self.sample_1b_path)
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    ctx = CryptContext.from_path(self.sample_1c_path, section='mypolicy', encoding='utf-16')
    self.assertEqual(ctx.to_dict(), self.sample_1_dict)
    self.assertRaises(EnvironmentError, CryptContext.from_path, os.path.join(here, 'sample1xxx.cfg'))
    self.assertRaises(NoSectionError, CryptContext.from_path, self.sample_1_path, section='fakesection')
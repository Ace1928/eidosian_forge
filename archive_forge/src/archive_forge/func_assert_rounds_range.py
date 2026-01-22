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
def assert_rounds_range(self, context, scheme, lower, upper):
    """helper to check vary_rounds covers specified range"""
    handler = context.handler(scheme)
    salt = handler.default_salt_chars[0:1] * handler.max_salt_size
    seen = set()
    for i in irange(300):
        h = context.genconfig(scheme, salt=salt)
        r = handler.from_string(h).rounds
        seen.add(r)
    self.assertEqual(min(seen), lower, 'vary_rounds had wrong lower limit:')
    self.assertEqual(max(seen), upper, 'vary_rounds had wrong upper limit:')
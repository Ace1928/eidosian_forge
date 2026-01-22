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
def test_35_to_string(self):
    """test to_string() method"""
    ctx = CryptContext(**self.sample_1_dict)
    dump = ctx.to_string()
    if not PY26:
        self.assertEqual(dump, self.sample_1_unicode)
    ctx2 = CryptContext.from_string(dump)
    self.assertEqual(ctx2.to_dict(), self.sample_1_dict)
    other = ctx.to_string(section='password-security')
    self.assertEqual(other, dump.replace('[passlib]', '[password-security]'))
    from passlib.tests.test_utils_handlers import UnsaltedHash
    ctx3 = CryptContext([UnsaltedHash, 'md5_crypt'])
    dump = ctx3.to_string()
    self.assertRegex(dump, "# NOTE: the 'unsalted_test_hash' handler\\(s\\) are not registered with Passlib")
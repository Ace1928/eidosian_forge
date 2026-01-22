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
def test_11_load_rollback(self):
    """test load() errors restore old state"""
    cc = CryptContext(['des_crypt', 'sha256_crypt'], sha256_crypt__default_rounds=5000, all__vary_rounds=0.1)
    result = cc.to_string()
    self.assertRaises(TypeError, cc.update, too__many__key__parts=True)
    self.assertEqual(cc.to_string(), result)
    self.assertRaises(KeyError, cc.update, fake_context_option=True)
    self.assertEqual(cc.to_string(), result)
    self.assertRaises(ValueError, cc.update, sha256_crypt__min_rounds=10000)
    self.assertEqual(cc.to_string(), result)
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
def test_44_identify(self):
    """test identify() border cases"""
    handlers = ['md5_crypt', 'des_crypt', 'bsdi_crypt']
    cc = CryptContext(handlers, bsdi_crypt__default_rounds=5)
    self.assertEqual(cc.identify('$9$232323123$1287319827'), None)
    self.assertRaises(ValueError, cc.identify, '$9$232323123$1287319827', required=True)
    cc = CryptContext(['des_crypt'])
    for hash, kwds in self.nonstring_vectors:
        self.assertRaises(TypeError, cc.identify, hash, **kwds)
    cc = CryptContext()
    self.assertIs(cc.identify('hash'), None)
    self.assertRaises(KeyError, cc.identify, 'hash', required=True)
    self.assertRaises(TypeError, cc.identify, None, category=1)
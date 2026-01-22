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
def test_45_verify(self):
    """test verify() scheme kwd"""
    handlers = ['md5_crypt', 'des_crypt', 'bsdi_crypt']
    cc = CryptContext(handlers, bsdi_crypt__default_rounds=5)
    h = hash.md5_crypt.hash('test')
    self.assertTrue(cc.verify('test', h))
    self.assertTrue(not cc.verify('notest', h))
    self.assertTrue(cc.verify('test', h, scheme='md5_crypt'))
    self.assertTrue(not cc.verify('notest', h, scheme='md5_crypt'))
    self.assertRaises(ValueError, cc.verify, 'test', h, scheme='bsdi_crypt')
    self.assertRaises(ValueError, cc.verify, 'stub', '$6$232323123$1287319827')
    cc = CryptContext(['des_crypt'])
    h = refhash = cc.hash('stub')
    for secret, kwds in self.nonstring_vectors:
        self.assertRaises(TypeError, cc.verify, secret, h, **kwds)
    self.assertFalse(cc.verify(secret, None))
    cc = CryptContext(['des_crypt'])
    for h, kwds in self.nonstring_vectors:
        if h is None:
            continue
        self.assertRaises(TypeError, cc.verify, 'secret', h, **kwds)
    self.assertRaises(KeyError, CryptContext().verify, 'secret', 'hash')
    self.assertRaises(KeyError, cc.verify, 'secret', refhash, scheme='fake')
    self.assertRaises(TypeError, cc.verify, 'secret', refhash, scheme=1)
    self.assertRaises(TypeError, cc.verify, 'secret', refhash, category=1)
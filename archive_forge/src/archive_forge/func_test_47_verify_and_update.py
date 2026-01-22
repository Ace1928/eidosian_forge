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
def test_47_verify_and_update(self):
    """test verify_and_update()"""
    cc = CryptContext(**self.sample_4_dict)
    h1 = cc.handler('des_crypt').hash('password')
    h2 = cc.handler('sha256_crypt').hash('password')
    ok, new_hash = cc.verify_and_update('wrongpass', h1)
    self.assertFalse(ok)
    self.assertIs(new_hash, None)
    ok, new_hash = cc.verify_and_update('wrongpass', h2)
    self.assertFalse(ok)
    self.assertIs(new_hash, None)
    ok, new_hash = cc.verify_and_update('password', h1)
    self.assertTrue(ok)
    self.assertTrue(cc.identify(new_hash), 'sha256_crypt')
    ok, new_hash = cc.verify_and_update('password', h2)
    self.assertTrue(ok)
    self.assertIs(new_hash, None)
    cc = CryptContext(['des_crypt'])
    hash = refhash = cc.hash('stub')
    for secret, kwds in self.nonstring_vectors:
        self.assertRaises(TypeError, cc.verify_and_update, secret, hash, **kwds)
    self.assertEqual(cc.verify_and_update(secret, None), (False, None))
    cc = CryptContext(['des_crypt'])
    for hash, kwds in self.nonstring_vectors:
        if hash is None:
            continue
        self.assertRaises(TypeError, cc.verify_and_update, 'secret', hash, **kwds)
    self.assertRaises(KeyError, CryptContext().verify_and_update, 'secret', 'hash')
    self.assertRaises(KeyError, cc.verify_and_update, 'secret', refhash, scheme='fake')
    self.assertRaises(TypeError, cc.verify_and_update, 'secret', refhash, scheme=1)
    self.assertRaises(TypeError, cc.verify_and_update, 'secret', refhash, category=1)
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
def test_48_context_kwds(self):
    """hash(), verify(), and verify_and_update() -- discard unused context keywords"""
    from passlib.hash import des_crypt, md5_crypt, postgres_md5
    des_hash = des_crypt.hash('stub')
    pg_root_hash = postgres_md5.hash('stub', user='root')
    pg_admin_hash = postgres_md5.hash('stub', user='admin')
    cc1 = CryptContext([des_crypt, md5_crypt])
    self.assertEqual(cc1.context_kwds, set())
    self.assertTrue(des_crypt.identify(cc1.hash('stub')), 'des_crypt')
    self.assertTrue(cc1.verify('stub', des_hash))
    self.assertEqual(cc1.verify_and_update('stub', des_hash), (True, None))
    with self.assertWarningList(['passing settings to.*is deprecated']):
        self.assertRaises(TypeError, cc1.hash, 'stub', user='root')
    self.assertRaises(TypeError, cc1.verify, 'stub', des_hash, user='root')
    self.assertRaises(TypeError, cc1.verify_and_update, 'stub', des_hash, user='root')
    cc2 = CryptContext([des_crypt, postgres_md5])
    self.assertEqual(cc2.context_kwds, set(['user']))
    self.assertTrue(des_crypt.identify(cc2.hash('stub')), 'des_crypt')
    self.assertTrue(cc2.verify('stub', des_hash))
    self.assertEqual(cc2.verify_and_update('stub', des_hash), (True, None))
    self.assertTrue(des_crypt.identify(cc2.hash('stub', user='root')), 'des_crypt')
    self.assertTrue(cc2.verify('stub', des_hash, user='root'))
    self.assertEqual(cc2.verify_and_update('stub', des_hash, user='root'), (True, None))
    with self.assertWarningList(['passing settings to.*is deprecated']):
        self.assertRaises(TypeError, cc2.hash, 'stub', badkwd='root')
    self.assertRaises(TypeError, cc2.verify, 'stub', des_hash, badkwd='root')
    self.assertRaises(TypeError, cc2.verify_and_update, 'stub', des_hash, badkwd='root')
    cc3 = CryptContext([postgres_md5, des_crypt], deprecated='auto')
    self.assertEqual(cc3.context_kwds, set(['user']))
    self.assertRaises(TypeError, cc3.hash, 'stub')
    self.assertRaises(TypeError, cc3.verify, 'stub', pg_root_hash)
    self.assertRaises(TypeError, cc3.verify_and_update, 'stub', pg_root_hash)
    self.assertEqual(cc3.hash('stub', user='root'), pg_root_hash)
    self.assertTrue(cc3.verify('stub', pg_root_hash, user='root'))
    self.assertEqual(cc3.verify_and_update('stub', pg_root_hash, user='root'), (True, None))
    self.assertEqual(cc3.verify_and_update('stub', pg_root_hash, user='admin'), (False, None))
    self.assertEqual(cc3.verify_and_update('stub', des_hash, user='root'), (True, pg_root_hash))
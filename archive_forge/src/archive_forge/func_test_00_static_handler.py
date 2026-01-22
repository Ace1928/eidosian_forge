from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
def test_00_static_handler(self):
    """test StaticHandler class"""

    class d1(uh.StaticHandler):
        name = 'd1'
        context_kwds = ('flag',)
        _hash_prefix = u('_')
        checksum_chars = u('ab')
        checksum_size = 1

        def __init__(self, flag=False, **kwds):
            super(d1, self).__init__(**kwds)
            self.flag = flag

        def _calc_checksum(self, secret):
            return u('b') if self.flag else u('a')
    self.assertTrue(d1.identify(u('_a')))
    self.assertTrue(d1.identify(b'_a'))
    self.assertTrue(d1.identify(u('_b')))
    self.assertFalse(d1.identify(u('_c')))
    self.assertFalse(d1.identify(b'_c'))
    self.assertFalse(d1.identify(u('a')))
    self.assertFalse(d1.identify(u('b')))
    self.assertFalse(d1.identify(u('c')))
    self.assertRaises(TypeError, d1.identify, None)
    self.assertRaises(TypeError, d1.identify, 1)
    self.assertEqual(d1.genconfig(), d1.hash(''))
    self.assertTrue(d1.verify('s', b'_a'))
    self.assertTrue(d1.verify('s', u('_a')))
    self.assertFalse(d1.verify('s', b'_b'))
    self.assertFalse(d1.verify('s', u('_b')))
    self.assertTrue(d1.verify('s', b'_b', flag=True))
    self.assertRaises(ValueError, d1.verify, 's', b'_c')
    self.assertRaises(ValueError, d1.verify, 's', u('_c'))
    self.assertEqual(d1.hash('s'), '_a')
    self.assertEqual(d1.hash('s', flag=True), '_b')
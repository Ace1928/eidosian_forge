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
def test_10_wrapped_attributes(self):
    d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}')
    self.assertEqual(d1.name, 'd1')
    self.assertIs(d1.setting_kwds, ldap_md5.setting_kwds)
    self.assertFalse('max_rounds' in dir(d1))
    d2 = uh.PrefixWrapper('d2', 'sha256_crypt', '{XXX}')
    self.assertIs(d2.setting_kwds, sha256_crypt.setting_kwds)
    self.assertTrue('max_rounds' in dir(d2))
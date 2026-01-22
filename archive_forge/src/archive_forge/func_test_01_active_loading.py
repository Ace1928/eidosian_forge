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
def test_01_active_loading(self):
    """test PrefixWrapper active loading of handler"""
    d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}')
    self.assertEqual(d1._wrapped_name, 'ldap_md5')
    self.assertIs(d1._wrapped_handler, ldap_md5)
    self.assertIs(d1.wrapped, ldap_md5)
    with dummy_handler_in_registry('ldap_md5') as dummy:
        self.assertIs(d1.wrapped, ldap_md5)
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
def test_14_bad_hash(self):
    """test orig_prefix sanity check"""
    h = uh.PrefixWrapper('h2', 'md5_crypt', orig_prefix='$6$')
    self.assertRaises(ValueError, h.hash, 'test')
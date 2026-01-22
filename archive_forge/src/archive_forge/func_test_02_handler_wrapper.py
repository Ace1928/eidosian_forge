from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
import sys
import re
from passlib import apps as _apps, exc, registry
from passlib.apps import django10_context, django14_context, django16_context
from passlib.context import CryptContext
from passlib.ext.django.utils import (
from passlib.utils.compat import iteritems, get_method_function, u
from passlib.utils.decor import memoized_property
from passlib.tests.utils import TestCase, TEST_MODE, handler_derived_from
from passlib.tests.test_handlers import get_handler_case
from passlib.hash import django_pbkdf2_sha256
def test_02_handler_wrapper(self):
    """test Hasher-compatible handler wrappers"""
    from django.contrib.auth import hashers
    passlib_to_django = DjangoTranslator().passlib_to_django
    if DJANGO_VERSION > (1, 10):
        self.assertRaises(ValueError, passlib_to_django, 'hex_md5')
    else:
        hasher = passlib_to_django('hex_md5')
        self.assertIsInstance(hasher, hashers.UnsaltedMD5PasswordHasher)
    hasher = passlib_to_django('django_bcrypt')
    self.assertIsInstance(hasher, hashers.BCryptPasswordHasher)
    from passlib.hash import sha256_crypt
    hasher = passlib_to_django('sha256_crypt')
    self.assertEqual(hasher.algorithm, 'passlib_sha256_crypt')
    encoded = hasher.encode('stub')
    self.assertTrue(sha256_crypt.verify('stub', encoded))
    self.assertTrue(hasher.verify('stub', encoded))
    self.assertFalse(hasher.verify('xxxx', encoded))
    encoded = hasher.encode('stub', 'abcd' * 4, rounds=1234)
    self.assertEqual(encoded, '$5$rounds=1234$abcdabcdabcdabcd$v2RWkZQzctPdejyRqmmTDQpZN6wTh7.RUy9zF2LftT6')
    self.assertEqual(hasher.safe_summary(encoded), {'algorithm': 'sha256_crypt', 'salt': u('abcdab**********'), 'rounds': 1234, 'hash': u('v2RWkZ*************************************')})
    self.assertRaises(KeyError, passlib_to_django, 'does_not_exist')
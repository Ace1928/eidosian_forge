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
def assert_valid_password(self, user, hash=UNSET, saved=None):
    """
        check that user object has a usable password hash.
        :param hash: optionally check it has this exact hash
        :param saved: check that mock commit history for user.password matches this list
        """
    if hash is UNSET:
        self.assertNotEqual(user.password, '!')
        self.assertNotEqual(user.password, None)
    else:
        self.assertEqual(user.password, hash)
    self.assertTrue(user.has_usable_password(), 'hash should be usable: %r' % (user.password,))
    self.assertEqual(user.pop_saved_passwords(), [] if saved is None else [saved])
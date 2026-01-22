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
def test_none_hash_value(self):
    """
        test how methods handle None as hash value
        """
    patched = self.patched
    from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
    user = FakeUser()
    user.password = None
    if quirks.none_causes_check_password_error and (not patched):
        self.assertRaises(TypeError, user.check_password, PASS1)
    else:
        self.assertFalse(user.check_password(PASS1))
    self.assertEqual(user.has_usable_password(), quirks.empty_is_usable_password)
    if quirks.none_causes_check_password_error and (not patched):
        self.assertRaises(TypeError, check_password, PASS1, None)
    else:
        self.assertFalse(check_password(PASS1, None))
    self.assertRaises(TypeError, identify_hasher, None)
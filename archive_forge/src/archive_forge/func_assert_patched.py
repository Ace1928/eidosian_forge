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
def assert_patched(self, context=None):
    """
        helper to ensure django HAS been patched, and is using specified config
        """
    mod = sys.modules.get('passlib.ext.django.models')
    self.assertTrue(mod and mod.adapter.patched, 'patch should have been enabled')
    for obj, attr, source, patched in self._iter_patch_candidates():
        if patched:
            self.assertTrue(source == 'passlib.ext.django.utils', 'obj=%r attr=%r should have been patched: %r' % (obj, attr, source))
        else:
            self.assertFalse(source.startswith('passlib.'), 'obj=%r attr=%r should not have been patched: %r' % (obj, attr, source))
    if context is not None:
        context = CryptContext._norm_source(context)
        self.assertEqual(mod.password_context.to_dict(resolve=True), context.to_dict(resolve=True))
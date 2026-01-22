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
def check_django_hasher_has_backend(name):
    """
    check whether django hasher is available;
    or if it should be skipped because django lacks third-party library.
    """
    assert name
    from django.contrib.auth.hashers import make_password
    try:
        make_password('', hasher=name)
        return True
    except ValueError as err:
        if re.match("Couldn't load '.*?' algorithm .* No module named .*", str(err)):
            return False
        raise
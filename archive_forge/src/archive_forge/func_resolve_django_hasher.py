from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def resolve_django_hasher(self, django_name, cached=True):
    """
        Take in a django algorithm name, return django hasher.
        """
    if hasattr(django_name, 'algorithm'):
        return django_name
    passlib_hasher = self.django_to_passlib(django_name, cached=cached)
    if django_name == 'unsalted_sha1' and passlib_hasher.name == 'django_salted_sha1':
        if not cached:
            return self._create_django_hasher(django_name)
        result = self._django_unsalted_sha1
        if result is None:
            result = self._django_unsalted_sha1 = self._create_django_hasher(django_name)
        return result
    return self.passlib_to_django(passlib_hasher, cached=cached)
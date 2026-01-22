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
def django_to_passlib(self, django_name, cached=True):
    """
        Convert Django hasher / name to Passlib hasher / name.
        If present, CryptContext will be checked instead of main registry.

        :param django_name:
            Django hasher class or algorithm name.
            "default" allowed if context provided.

        :raises ValueError:
            if can't resolve hasher.

        :returns:
            passlib hasher or name
        """
    if hasattr(django_name, 'algorithm'):
        if isinstance(django_name, _PasslibHasherWrapper):
            return django_name.passlib_handler
        django_name = django_name.algorithm
    if cached:
        cache = self._passlib_hasher_cache
        try:
            return cache[django_name]
        except KeyError:
            pass
        result = cache[django_name] = self.django_to_passlib(django_name, cached=False)
        return result
    if django_name.startswith(PASSLIB_WRAPPER_PREFIX):
        passlib_name = django_name[len(PASSLIB_WRAPPER_PREFIX):]
        return self._get_passlib_hasher(passlib_name)
    if django_name == 'default':
        context = self.context
        if context is None:
            raise TypeError("can't determine default scheme w/ context")
        return context.handler()
    if django_name == 'unsalted_sha1':
        django_name = 'sha1'
    context = self.context
    if context is None:
        candidates = (registry.get_crypt_handler(passlib_name) for passlib_name in registry.list_crypt_handlers() if passlib_name.startswith(DJANGO_COMPAT_PREFIX) or passlib_name in _other_django_hashes)
    else:
        candidates = context.schemes(resolve=True)
    for handler in candidates:
        if getattr(handler, 'django_name', None) == django_name:
            return handler
    raise ValueError("can't translate django name to passlib name: %r" % (django_name,))
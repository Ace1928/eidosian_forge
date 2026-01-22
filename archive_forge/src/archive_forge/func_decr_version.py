import time
import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def decr_version(self, key, delta=1, version=None):
    """
        Subtract delta from the cache version for the supplied key. Return the
        new version.
        """
    return self.incr_version(key, -delta, version)
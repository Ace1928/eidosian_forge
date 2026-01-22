import time
import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def incr_version(self, key, delta=1, version=None):
    """
        Add delta to the cache version for the supplied key. Return the new
        version.
        """
    if version is None:
        version = self.version
    value = self.get(key, self._missing_key, version=version)
    if value is self._missing_key:
        raise ValueError("Key '%s' not found" % key)
    self.set(key, value, version=version + delta)
    self.delete(key, version=version)
    return version + delta
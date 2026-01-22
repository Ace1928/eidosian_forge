from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
def expiring_lrucache(self, name=None, maxsize=None, timeout=None):
    'Named arguments:\n\n        - name (optional) is a string, and should be unique amongst all caches\n\n        - maxsize (optional) is an int, overriding any default value set by\n          the constructor\n\n        - timeout (optional) is an int, overriding any default value set by\n          the constructor or the default value (%d seconds)\n        ' % _DEFAULT_TIMEOUT
    name, maxsize, timeout = self._resolve_setting(name, maxsize, timeout)
    cache = self._cache[name] = ExpiringLRUCache(maxsize, timeout)
    return lru_cache(maxsize, cache, timeout)
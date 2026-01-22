from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
def lrucache(self, name=None, maxsize=None):
    """Named arguments:
        
        - name (optional) is a string, and should be unique amongst all caches

        - maxsize (optional) is an int, overriding any default value set by
          the constructor
        """
    name, maxsize, _ = self._resolve_setting(name, maxsize)
    cache = self._cache[name] = LRUCache(maxsize)
    return lru_cache(maxsize, cache)
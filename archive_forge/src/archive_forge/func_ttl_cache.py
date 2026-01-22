import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm with a per-item time-to-live (TTL) value.
    """
    if maxsize is None:
        return _cache(_UnboundTTLCache(ttl, timer), typed)
    elif callable(maxsize):
        return _cache(TTLCache(128, ttl, timer), typed)(maxsize)
    else:
        return _cache(TTLCache(maxsize, ttl, timer), typed)
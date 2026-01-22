import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
def mru_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Most Recently Used (MRU)
    algorithm.
    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(MRUCache(128), typed)(maxsize)
    else:
        return _cache(MRUCache(maxsize), typed)
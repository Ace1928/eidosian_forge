import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
def cached(cache, key=_defaultkey, lock=None):
    """Decorator to wrap a function with a memoizing callable that saves
    results in a cache.

    """

    def decorator(func):
        if cache is None:

            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
        elif lock is None:

            def wrapper(*args, **kwargs):
                k = key(*args, **kwargs)
                try:
                    return cache[k]
                except KeyError:
                    pass
                v = func(*args, **kwargs)
                try:
                    cache[k] = v
                except ValueError:
                    pass
                return v
        else:

            def wrapper(*args, **kwargs):
                k = key(*args, **kwargs)
                try:
                    with lock:
                        return cache[k]
                except KeyError:
                    pass
                v = func(*args, **kwargs)
                try:
                    with lock:
                        return cache.setdefault(k, v)
                except ValueError:
                    return v
        return functools.update_wrapper(wrapper, func)
    return decorator
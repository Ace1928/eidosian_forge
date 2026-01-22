import functools
import math
import os
import random
import threading
import time
from .core import ENOVAL, args_to_key, full_name
def memoize_stampede(cache, expire, name=None, typed=False, tag=None, beta=1, ignore=()):
    """Memoizing cache decorator with cache stampede protection.

    Cache stampedes are a type of system overload that can occur when parallel
    computing systems using memoization come under heavy load. This behaviour
    is sometimes also called dog-piling, cache miss storm, cache choking, or
    the thundering herd problem.

    The memoization decorator implements cache stampede protection through
    early recomputation. Early recomputation of function results will occur
    probabilistically before expiration in a background thread of
    execution. Early probabilistic recomputation is based on research by
    Vattani, A.; Chierichetti, F.; Lowenstein, K. (2015), Optimal Probabilistic
    Cache Stampede Prevention, VLDB, pp. 886-897, ISSN 2150-8097

    If name is set to None (default), the callable name will be determined
    automatically.

    If typed is set to True, function arguments of different types will be
    cached separately. For example, f(3) and f(3.0) will be treated as distinct
    calls with distinct results.

    The original underlying function is accessible through the `__wrapped__`
    attribute. This is useful for introspection, for bypassing the cache, or
    for rewrapping the function with a different cache.

    >>> from diskcache import Cache
    >>> cache = Cache()
    >>> @memoize_stampede(cache, expire=1)
    ... def fib(number):
    ...     if number == 0:
    ...         return 0
    ...     elif number == 1:
    ...         return 1
    ...     else:
    ...         return fib(number - 1) + fib(number - 2)
    >>> print(fib(100))
    354224848179261915075

    An additional `__cache_key__` attribute can be used to generate the cache
    key used for the given arguments.

    >>> key = fib.__cache_key__(100)
    >>> del cache[key]

    Remember to call memoize when decorating a callable. If you forget, then a
    TypeError will occur.

    :param cache: cache to store callable arguments and return values
    :param float expire: seconds until arguments expire
    :param str name: name given for callable (default None, automatic)
    :param bool typed: cache different types separately (default False)
    :param str tag: text to associate with arguments (default None)
    :param set ignore: positional or keyword args to ignore (default ())
    :return: callable decorator

    """

    def decorator(func):
        """Decorator created by memoize call for callable."""
        base = (full_name(func),) if name is None else (name,)

        def timer(*args, **kwargs):
            """Time execution of `func` and return result and time delta."""
            start = time.time()
            result = func(*args, **kwargs)
            delta = time.time() - start
            return (result, delta)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper for callable to cache arguments and return values."""
            key = wrapper.__cache_key__(*args, **kwargs)
            pair, expire_time = cache.get(key, default=ENOVAL, expire_time=True, retry=True)
            if pair is not ENOVAL:
                result, delta = pair
                now = time.time()
                ttl = expire_time - now
                if -delta * beta * math.log(random.random()) < ttl:
                    return result
                thread_key = key + (ENOVAL,)
                thread_added = cache.add(thread_key, None, expire=delta, retry=True)
                if thread_added:

                    def recompute():
                        with cache:
                            pair = timer(*args, **kwargs)
                            cache.set(key, pair, expire=expire, tag=tag, retry=True)
                    thread = threading.Thread(target=recompute)
                    thread.daemon = True
                    thread.start()
                return result
            pair = timer(*args, **kwargs)
            cache.set(key, pair, expire=expire, tag=tag, retry=True)
            return pair[0]

        def __cache_key__(*args, **kwargs):
            """Make key for cache given function arguments."""
            return args_to_key(base, args, kwargs, typed, ignore)
        wrapper.__cache_key__ = __cache_key__
        return wrapper
    return decorator
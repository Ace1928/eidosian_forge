import time
from functools import wraps
from typing import Any, Dict, Tuple
from jedi import settings
from parso.cache import parser_cache
def signature_time_cache(time_add_setting):
    """
    This decorator works as follows: Call it with a setting and after that
    use the function with a callable that returns the key.
    But: This function is only called if the key is not available. After a
    certain amount of time (`time_add_setting`) the cache is invalid.

    If the given key is None, the function will not be cached.
    """

    def _temp(key_func):
        dct = {}
        _time_caches[time_add_setting] = dct

        def wrapper(*args, **kwargs):
            generator = key_func(*args, **kwargs)
            key = next(generator)
            try:
                expiry, value = dct[key]
                if expiry > time.time():
                    return value
            except KeyError:
                pass
            value = next(generator)
            time_add = getattr(settings, time_add_setting)
            if key is not None:
                dct[key] = (time.time() + time_add, value)
            return value
        return wrapper
    return _temp
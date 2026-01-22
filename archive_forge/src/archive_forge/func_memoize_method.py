import time
from functools import wraps
from typing import Any, Dict, Tuple
from jedi import settings
from parso.cache import parser_cache
def memoize_method(method):
    """A normal memoize function."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        cache_dict = self.__dict__.setdefault('_memoize_method_dct', {})
        dct = cache_dict.setdefault(method, {})
        key = (args, frozenset(kwargs.items()))
        try:
            return dct[key]
        except KeyError:
            result = method(self, *args, **kwargs)
            dct[key] = result
            return result
    return wrapper
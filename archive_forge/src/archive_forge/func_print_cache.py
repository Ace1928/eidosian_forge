from importlib import import_module
from typing import Callable
from functools import lru_cache, wraps
def print_cache(self):
    """print cache info"""
    for item in self:
        name = item.__name__
        myfunc = item
        while hasattr(myfunc, '__wrapped__'):
            if hasattr(myfunc, 'cache_info'):
                info = myfunc.cache_info()
                break
            else:
                myfunc = myfunc.__wrapped__
        else:
            info = None
        print(name, info)
import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def init_once(self, func, tag):
    try:
        x = self._init_once_cache[tag]
    except KeyError:
        x = self._init_once_cache.setdefault(tag, (False, allocate_lock()))
    if x[0]:
        return x[1]
    with x[1]:
        x = self._init_once_cache[tag]
        if x[0]:
            return x[1]
        result = func()
        self._init_once_cache[tag] = (True, result)
    return result
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
def lru(maxsize=128):
    """Returns cached result if function was run with same args before.

  Wraps functools.lru_cache, so it's not referenced at import in Python 2 and
  unsupported Python 3 distributions.

  Args:
    maxsize (int|None): From Python functools docs: "...saves up to the maxsize
      most recent calls... If maxsize is set to None, the LRU feature is
      disabled and the cache can grow without bound."

  Returns:
    Wrapped functools.lru_cache.
  """

    def _wrapper(function):
        if getattr(functools, 'lru_cache', None):
            return functools.lru_cache(maxsize=maxsize)(function)
        return FakeLruCache(function)
    return _wrapper
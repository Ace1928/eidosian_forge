import operator as op
from collections import OrderedDict
from collections.abc import (
from contextlib import contextmanager
from shutil import rmtree
from .core import ENOVAL, Cache
@classmethod
def fromcache(cls, cache, *args, **kwargs):
    """Initialize index using `cache` and update items.

        >>> cache = Cache()
        >>> index = Index.fromcache(cache, {'a': 1, 'b': 2, 'c': 3})
        >>> index.cache is cache
        True
        >>> len(index)
        3
        >>> 'b' in index
        True
        >>> index['c']
        3

        :param Cache cache: cache to use
        :param args: mapping or sequence of items
        :param kwargs: mapping of items
        :return: initialized Index

        """
    self = cls.__new__(cls)
    self._cache = cache
    self._update(*args, **kwargs)
    return self
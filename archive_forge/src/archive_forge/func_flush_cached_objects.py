from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
def flush_cached_objects(self, force_all: bool=False) -> Generator[U, None, None]:
    """Return a generator over cached objects evicted from the cache.

        This method yields all cached objects that should be evicted from the
        cache for cleanup by the caller.

        If the number of max objects is lower than the number of
        cached objects for a given key, objects are evicted until
        the numbers are equal.

        If `max_keep_one=True` (and ``force_all=False``), one cached object
        may be retained.

        Objects are evicted FIFO.

        If ``force_all=True``, all objects are evicted.

        Args:
            force_all: If True, all objects are flushed. This takes precedence
                over ``keep_one``.

        Yields:
            Evicted objects to be cleaned up by caller.

        """
    keep_one = self._may_keep_one and (not force_all)
    for key, objs in self._cached_objects.items():
        max_cached = self._max_num_objects[key] if not force_all else 0
        if self._num_cached_objects == 1 and keep_one and (not any((v for v in self._max_num_objects.values()))):
            break
        while len(objs) > max_cached:
            self._num_cached_objects -= 1
            yield objs.pop(0)
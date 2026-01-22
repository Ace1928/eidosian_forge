from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
def cache_object(self, key: T, obj: U) -> bool:
    """Cache object for a given key.

        This will put the object into a cache, assuming the number
        of cached objects for this key is less than the number of
        max objects for this key.

        An exception is made if `max_keep_one=True` and no other
        objects are cached globally. In that case, the object can
        still be cached.

        Args:
            key: Group key.
            obj: Object to cache.

        Returns:
            True if the object has been cached. False otherwise.

        """
    if len(self._cached_objects[key]) >= self._max_num_objects[key]:
        if not self._may_keep_one:
            return False
        if self._num_cached_objects > 0:
            return False
        if any((v for v in self._max_num_objects.values())):
            return False
    self._cached_objects[key].append(obj)
    self._num_cached_objects += 1
    return True
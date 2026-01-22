import abc
import weakref
from numba.core import errors
def load_cache(self, orig_disp):
    """Load a dispatcher associated with the given key.
        """
    out = self._cache.get(orig_disp)
    if out is None:
        self._stat_miss += 1
    else:
        self._stat_hit += 1
    return out
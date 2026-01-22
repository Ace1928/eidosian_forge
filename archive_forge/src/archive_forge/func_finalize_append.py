import numbers
from functools import reduce
from operator import mul
import numpy as np
def finalize_append(self):
    """Finalize process of appending several elements to `self`

        :meth:`append` can be a lot faster if it knows that it is appending
        several elements instead of a single element.  To tell the append
        method this is the case, use ``cache_build=True``.  This method
        finalizes the series of append operations after a call to
        :meth:`append` with ``cache_build=True``.
        """
    if self._build_cache is None:
        return
    self._build_cache.update_seq(self)
    self._build_cache = None
    self.shrink_data()
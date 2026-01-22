from contextlib import contextmanager
import numpy as np
from_record_like = None
def copy_to_host(self, ary=None, stream=0):
    if ary is None:
        ary = np.empty_like(self._ary)
    else:
        check_array_compatibility(self, ary)
    np.copyto(ary, self._ary)
    return ary
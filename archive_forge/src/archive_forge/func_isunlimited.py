import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
def isunlimited(self):
    """Return ``True`` if dimension is unlimited, otherwise ``False``."""
    if self._phony:
        return False
    return self._h5ds.maxshape == (None,)
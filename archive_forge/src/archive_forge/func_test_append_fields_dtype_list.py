import os
import numpy as np
from numpy.testing import (
def test_append_fields_dtype_list(self):
    from numpy.lib.recfunctions import append_fields
    base = np.array([1, 2, 3], dtype=np.int32)
    names = ['a', 'b', 'c']
    data = np.eye(3).astype(np.int32)
    dlist = [np.float64, np.int32, np.int32]
    try:
        append_fields(base, names, data, dlist)
    except Exception:
        raise AssertionError()
from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
@numba.jit(nopython=True, nogil=True, parallel=True)
def nan_agg_with_axis(table):
    result = np.empty(table.shape[1])
    for i in numba.prange(table.shape[1]):
        partition = table[:, i]
        result[i] = nan_func(partition)
    return result
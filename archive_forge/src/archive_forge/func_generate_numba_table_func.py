from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
@functools.cache
def generate_numba_table_func(func: Callable[..., np.ndarray], nopython: bool, nogil: bool, parallel: bool):
    """
    Generate a numba jitted function to apply window calculations table-wise.

    Func will be passed a M window size x N number of columns array, and
    must return a 1 x N number of columns array. Func is intended to operate
    row-wise, but the result will be transposed for axis=1.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Parameters
    ----------
    func : function
        function to be applied to each window and will be JITed
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_table(values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int, *args: Any):
        result = np.empty((len(begin), values.shape[1]))
        min_periods_mask = np.empty(result.shape)
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window), axis=0)
            sub_result = numba_func(window, *args)
            nan_mask = len(window) - count_nan >= minimum_periods
            min_periods_mask[i, :] = nan_mask
            result[i, :] = sub_result
        result = np.where(min_periods_mask, result, np.nan)
        return result
    return roll_table
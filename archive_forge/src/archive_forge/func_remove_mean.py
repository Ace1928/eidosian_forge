from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
from pandas.core._numba.kernels.sum_ import grouped_kahan_sum
@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_mean(val: float, nobs: int, sum_x: float, neg_ct: int, compensation: float) -> tuple[int, float, int, float]:
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct -= 1
    return (nobs, sum_x, neg_ct, compensation)
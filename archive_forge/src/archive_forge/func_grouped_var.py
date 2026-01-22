from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_var(values: np.ndarray, result_dtype: np.dtype, labels: npt.NDArray[np.intp], ngroups: int, min_periods: int, ddof: int=1) -> tuple[np.ndarray, list[int]]:
    N = len(labels)
    nobs_arr = np.zeros(ngroups, dtype=np.int64)
    comp_arr = np.zeros(ngroups, dtype=values.dtype)
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)
    prev_vals = np.zeros(ngroups, dtype=values.dtype)
    output = np.zeros(ngroups, dtype=result_dtype)
    means = np.zeros(ngroups, dtype=result_dtype)
    for i in range(N):
        lab = labels[i]
        val = values[i]
        if lab < 0:
            continue
        mean_x = means[lab]
        ssqdm_x = output[lab]
        nobs = nobs_arr[lab]
        compensation_add = comp_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]
        nobs, mean_x, ssqdm_x, compensation_add, num_consecutive_same_value, prev_value = add_var(val, nobs, mean_x, ssqdm_x, compensation_add, num_consecutive_same_value, prev_value)
        output[lab] = ssqdm_x
        means[lab] = mean_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs
    for lab in range(ngroups):
        nobs = nobs_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        ssqdm_x = output[lab]
        if nobs >= min_periods and nobs > ddof:
            if nobs == 1 or num_consecutive_same_value >= nobs:
                result = 0.0
            else:
                result = ssqdm_x / (nobs - ddof)
        else:
            result = np.nan
        output[lab] = result
    na_pos = [0 for i in range(0)]
    return (output, na_pos)
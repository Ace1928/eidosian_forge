from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np
@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_min_max(values: np.ndarray, result_dtype: np.dtype, labels: npt.NDArray[np.intp], ngroups: int, min_periods: int, is_max: bool) -> tuple[np.ndarray, list[int]]:
    N = len(labels)
    nobs = np.zeros(ngroups, dtype=np.int64)
    na_pos = []
    output = np.empty(ngroups, dtype=result_dtype)
    for i in range(N):
        lab = labels[i]
        val = values[i]
        if lab < 0:
            continue
        if values.dtype.kind == 'i' or not np.isnan(val):
            nobs[lab] += 1
        else:
            continue
        if nobs[lab] == 1:
            output[lab] = val
            continue
        if is_max:
            if val > output[lab]:
                output[lab] = val
        elif val < output[lab]:
            output[lab] = val
    for lab, count in enumerate(nobs):
        if count < min_periods:
            na_pos.append(lab)
    return (output, na_pos)
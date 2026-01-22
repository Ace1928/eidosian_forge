from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_integer_dtype
from tlz import merge, merge_sorted, take
from dask.base import tokenize
from dask.dataframe.core import Series
from dask.dataframe.dispatch import tolist_dispatch
from dask.utils import is_cupy_type, random_state_data
def process_val_weights(vals_and_weights, npartitions, dtype_info):
    """Calculate final approximate percentiles given weighted vals

    ``vals_and_weights`` is assumed to be sorted.  We take a cumulative
    sum of the weights, which makes them percentile-like (their scale is
    [0, N] instead of [0, 100]).  Next we find the divisions to create
    partitions of approximately equal size.

    It is possible for adjacent values of the result to be the same.  Since
    these determine the divisions of the new partitions, some partitions
    may be empty.  This can happen if we under-sample the data, or if there
    aren't enough unique values in the column.  Increasing ``upsample``
    keyword argument in ``df.set_index`` may help.
    """
    dtype, info = dtype_info
    if not vals_and_weights:
        try:
            return np.array(None, dtype=dtype)
        except Exception:
            return np.array(None, dtype=np.float64)
    vals, weights = vals_and_weights
    vals = np.array(vals)
    weights = np.array(weights)
    if len(vals) == npartitions + 1:
        rv = vals
    elif len(vals) < npartitions + 1:
        if np.issubdtype(vals.dtype, np.number) and (not isinstance(dtype, pd.CategoricalDtype)):
            q_weights = np.cumsum(weights)
            q_target = np.linspace(q_weights[0], q_weights[-1], npartitions + 1)
            rv = np.interp(q_target, q_weights, vals)
        else:
            duplicated_index = np.linspace(0, len(vals) - 1, npartitions - len(vals) + 1, dtype=int)
            duplicated_vals = vals[duplicated_index]
            rv = np.concatenate([vals, duplicated_vals])
            rv.sort()
    else:
        target_weight = weights.sum() / npartitions
        jumbo_mask = weights >= target_weight
        jumbo_vals = vals[jumbo_mask]
        trimmed_vals = vals[~jumbo_mask]
        trimmed_weights = weights[~jumbo_mask]
        trimmed_npartitions = npartitions - len(jumbo_vals)
        q_weights = np.cumsum(trimmed_weights)
        q_target = np.linspace(0, q_weights[-1], trimmed_npartitions + 1)
        left = np.searchsorted(q_weights, q_target, side='left')
        right = np.searchsorted(q_weights, q_target, side='right') - 1
        np.maximum(right, 0, right)
        lower = np.minimum(left, right)
        trimmed = trimmed_vals[lower]
        rv = np.concatenate([trimmed, jumbo_vals])
        rv.sort()
    if isinstance(dtype, pd.CategoricalDtype):
        rv = pd.Categorical.from_codes(rv, info[0], info[1])
    elif isinstance(dtype, pd.DatetimeTZDtype):
        rv = pd.DatetimeIndex(rv).tz_localize(dtype.tz)
    elif 'datetime64' in str(dtype):
        rv = pd.DatetimeIndex(rv, dtype=dtype)
    elif rv.dtype != dtype:
        if is_integer_dtype(dtype) and pd.api.types.is_float_dtype(rv.dtype):
            rv = np.floor(rv)
        rv = pd.array(rv, dtype=dtype)
    return rv
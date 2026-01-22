from __future__ import annotations
import warnings
from collections.abc import Iterator
from functools import wraps
from numbers import Number
import numpy as np
from tlz import merge
from dask.array.core import Array
from dask.array.numpy_compat import NUMPY_GE_122
from dask.array.numpy_compat import percentile as np_percentile
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
def merge_percentiles(finalq, qs, vals, method='lower', Ns=None, raise_on_nan=True):
    """Combine several percentile calculations of different data.

    Parameters
    ----------

    finalq : numpy.array
        Percentiles to compute (must use same scale as ``qs``).
    qs : sequence of :class:`numpy.array`s
        Percentiles calculated on different sets of data.
    vals : sequence of :class:`numpy.array`s
        Resulting values associated with percentiles ``qs``.
    Ns : sequence of integers
        The number of data elements associated with each data set.
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        Specify the interpolation method to use to calculate final
        percentiles.  For more information, see :func:`numpy.percentile`.

    Examples
    --------

    >>> finalq = [10, 20, 30, 40, 50, 60, 70, 80]
    >>> qs = [[20, 40, 60, 80], [20, 40, 60, 80]]
    >>> vals = [np.array([1, 2, 3, 4]), np.array([10, 11, 12, 13])]
    >>> Ns = [100, 100]  # Both original arrays had 100 elements

    >>> merge_percentiles(finalq, qs, vals, Ns=Ns)
    array([ 1,  2,  3,  4, 10, 11, 12, 13])
    """
    from dask.array.utils import array_safe
    if isinstance(finalq, Iterator):
        finalq = list(finalq)
    finalq = array_safe(finalq, like=finalq)
    qs = list(map(list, qs))
    vals = list(vals)
    if Ns is None:
        vals, Ns = zip(*vals)
    Ns = list(Ns)
    L = list(zip(*[(q, val, N) for q, val, N in zip(qs, vals, Ns) if N]))
    if not L:
        if raise_on_nan:
            raise ValueError('No non-trivial arrays found')
        return np.full(len(qs[0]) - 2, np.nan)
    qs, vals, Ns = L
    if vals[0].dtype.name == 'category':
        result = merge_percentiles(finalq, qs, [v.codes for v in vals], method, Ns, raise_on_nan)
        import pandas as pd
        return pd.Categorical.from_codes(result, vals[0].categories, vals[0].ordered)
    if not np.issubdtype(vals[0].dtype, np.number):
        method = 'nearest'
    if len(vals) != len(qs) or len(Ns) != len(qs):
        raise ValueError('qs, vals, and Ns parameters must be the same length')
    counts = []
    for q, N in zip(qs, Ns):
        count = np.empty_like(finalq, shape=len(q))
        count[1:] = np.diff(array_safe(q, like=q[0]))
        count[0] = q[0]
        count *= N
        counts.append(count)
    combined_vals = np.concatenate(vals)
    combined_counts = array_safe(np.concatenate(counts), like=combined_vals)
    sort_order = np.argsort(combined_vals)
    combined_vals = np.take(combined_vals, sort_order)
    combined_counts = np.take(combined_counts, sort_order)
    combined_q = np.cumsum(combined_counts)
    finalq = array_safe(finalq, like=combined_vals)
    desired_q = finalq * sum(Ns)
    if method == 'linear':
        rv = np.interp(desired_q, combined_q, combined_vals)
    else:
        left = np.searchsorted(combined_q, desired_q, side='left')
        right = np.searchsorted(combined_q, desired_q, side='right') - 1
        np.minimum(left, len(combined_vals) - 1, left)
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        if method == 'lower':
            rv = combined_vals[lower]
        elif method == 'higher':
            rv = combined_vals[upper]
        elif method == 'midpoint':
            rv = 0.5 * (combined_vals[lower] + combined_vals[upper])
        elif method == 'nearest':
            lower_residual = np.abs(combined_q[lower] - desired_q)
            upper_residual = np.abs(combined_q[upper] - desired_q)
            mask = lower_residual > upper_residual
            index = lower
            index[mask] = upper[mask]
            rv = combined_vals[index]
        else:
            raise ValueError("interpolation method can only be 'linear', 'lower', 'higher', 'midpoint', or 'nearest'")
    return rv
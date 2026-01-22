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
Combine several percentile calculations of different data.

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
    
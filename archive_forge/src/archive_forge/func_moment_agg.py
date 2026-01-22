from __future__ import annotations
import builtins
import contextlib
import math
import operator
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import product, repeat
from numbers import Integral, Number
import numpy as np
from tlz import accumulate, compose, drop, get, partition_all, pluck
from dask import config
from dask.array import chunk
from dask.array.blockwise import blockwise
from dask.array.core import (
from dask.array.creation import arange, diagonal
from dask.array.dispatch import divide_lookup, nannumel_lookup, numel_lookup
from dask.array.numpy_compat import ComplexWarning
from dask.array.utils import (
from dask.array.wrap import ones, zeros
from dask.base import tokenize
from dask.blockwise import lol_tuples
from dask.highlevelgraph import HighLevelGraph
from dask.utils import (
def moment_agg(pairs, order=2, ddof=0, dtype='f8', sum=np.sum, axis=None, computing_meta=False, **kwargs):
    if not isinstance(pairs, list):
        pairs = [pairs]
    kwargs['dtype'] = dtype
    keepdim_kw = kwargs.copy()
    keepdim_kw['keepdims'] = True
    keepdim_kw['dtype'] = None
    ns = deepmap(lambda pair: pair['n'], pairs) if not computing_meta else pairs
    ns = _concatenate2(ns, axes=axis)
    n = ns.sum(axis=axis, **keepdim_kw)
    if computing_meta:
        return n
    totals = _concatenate2(deepmap(lambda pair: pair['total'], pairs), axes=axis)
    Ms = _concatenate2(deepmap(lambda pair: pair['M'], pairs), axes=axis)
    mu = divide(totals.sum(axis=axis, **keepdim_kw), n)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.issubdtype(totals.dtype, np.complexfloating):
            inner_term = np.abs(divide(totals, ns) - mu)
        else:
            inner_term = divide(totals, ns, dtype=dtype) - mu
    M = _moment_helper(Ms, ns, inner_term, order, sum, axis, kwargs)
    denominator = n.sum(axis=axis, **kwargs) - ddof
    if isinstance(denominator, Number):
        if denominator < 0:
            denominator = np.nan
    elif denominator is not np.ma.masked:
        denominator[denominator < 0] = np.nan
    return divide(M, denominator, dtype=dtype)
from __future__ import annotations
from functools import wraps
import numpy as np
from dask.array import chunk
from dask.array.core import asanyarray, blockwise, elemwise, map_blocks
from dask.array.reductions import reduction
from dask.array.routines import _average
from dask.array.routines import nonzero as _nonzero
from dask.base import normalize_token
from dask.utils import derived_from
@derived_from(np.ma)
def masked_equal(a, value):
    a = asanyarray(a)
    if getattr(value, 'shape', ()):
        raise ValueError("da.ma.masked_equal doesn't support array `value`s")
    inds = tuple(range(a.ndim))
    return blockwise(np.ma.masked_equal, inds, a, inds, value, (), dtype=a.dtype)
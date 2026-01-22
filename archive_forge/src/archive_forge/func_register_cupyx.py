from __future__ import annotations
import math
import numpy as np
from dask.array import chunk
from dask.array.core import Array
from dask.array.dispatch import (
from dask.array.numpy_compat import divide as np_divide
from dask.array.numpy_compat import ma_divide
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
@tensordot_lookup.register_lazy('cupyx')
@concatenate_lookup.register_lazy('cupyx')
def register_cupyx():
    from cupyx.scipy.sparse import spmatrix
    try:
        from cupyx.scipy.sparse import hstack, vstack
    except ImportError as e:
        raise ImportError('Stacking of sparse arrays requires at least CuPy version 8.0.0') from e

    def _concat_cupy_sparse(L, axis=0):
        if axis == 0:
            return vstack(L)
        elif axis == 1:
            return hstack(L)
        else:
            msg = 'Can only concatenate cupy sparse matrices for axis in {0, 1}.  Got %s' % axis
            raise ValueError(msg)
    concatenate_lookup.register(spmatrix, _concat_cupy_sparse)
    tensordot_lookup.register(spmatrix, _tensordot_scipy_sparse)
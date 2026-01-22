from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def test_unravel_index():
    rng = np.random.default_rng()
    for nindices, shape, order in [(0, (15,), 'C'), (1, (15,), 'C'), (3, (15,), 'C'), (3, (15,), 'F'), (2, (15, 16), 'C'), (2, (15, 16), 'F')]:
        arr = rng.random(shape)
        darr = da.from_array(arr, chunks=1)
        findices = rng.integers(np.prod(shape, dtype=int), size=nindices)
        d_findices = da.from_array(findices, chunks=1)
        indices = np.unravel_index(findices, shape, order)
        d_indices = da.unravel_index(d_findices, shape, order)
        assert isinstance(d_indices, type(indices))
        assert len(d_indices) == len(indices)
        for i in range(len(indices)):
            assert_eq(d_indices[i], indices[i])
        assert_eq(darr.vindex[dask.compute(*d_indices)], arr[indices])
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
def test_unravel_index_empty():
    shape = tuple()
    findices = np.array(0, dtype=int)
    d_findices = da.from_array(findices, chunks=1)
    indices = np.unravel_index(findices, shape)
    d_indices = da.unravel_index(d_findices, shape)
    assert isinstance(d_indices, type(indices))
    assert len(d_indices) == len(indices) == 0
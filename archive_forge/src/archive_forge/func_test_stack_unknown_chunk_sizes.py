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
@pytest.mark.parametrize('np_func,dsk_func,nan_chunk', [(np.hstack, da.hstack, 0), (np.dstack, da.dstack, 1), (np.vstack, da.vstack, 2)])
def test_stack_unknown_chunk_sizes(np_func, dsk_func, nan_chunk):
    shape = (100, 100, 100)
    x = da.ones(shape, chunks=(50, 50, 50))
    y = np.ones(shape)
    tmp = list(x._chunks)
    tmp[nan_chunk] = (np.nan,) * 2
    x._chunks = tuple(tmp)
    with pytest.raises(ValueError):
        dsk_func((x, x))
    np_stacked = np_func((y, y))
    dsk_stacked = dsk_func((x, x), allow_unknown_chunksizes=True)
    assert_eq(np_stacked, dsk_stacked)
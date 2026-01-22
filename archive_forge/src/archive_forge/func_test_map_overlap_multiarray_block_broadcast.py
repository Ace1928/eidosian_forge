from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_multiarray_block_broadcast():

    def func(x, y):
        z = x.size + y.size
        return np.ones((3, 3)) * z
    x = da.ones((12,), chunks=12)
    y = da.ones((16, 12), chunks=(8, 6))
    z = da.map_overlap(func, x, y, chunks=(3, 3), depth=1, trim=True, boundary='reflect')
    assert_eq(z, z)
    assert z.shape == (2, 2)
    assert_eq(z.sum(), 4.0 * (10 * 8 + 8))
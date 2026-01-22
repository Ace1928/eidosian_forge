from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_multiarray_different_depths():
    x = da.ones(5, dtype='int')
    y = da.ones(5, dtype='int')

    def run(depth):
        return da.map_overlap(lambda x, y: x.sum() + y.sum(), x, y, depth=depth, chunks=(0,), trim=False, boundary='reflect').compute()
    assert run([0, 0]) == 10
    assert run([0, 1]) == 12
    assert run([1, 1]) == 14
    assert run([1, 2]) == 16
    assert run([0, 5]) == 20
    assert run([5, 5]) == 30
    with pytest.raises(ValueError):
        run([0, 6])
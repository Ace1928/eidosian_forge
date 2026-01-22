from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_nearest_overlap():
    a = np.arange(144).reshape(12, 12).astype(float)
    darr = da.from_array(a, chunks=(6, 6))
    garr = overlap(darr, depth={0: 5, 1: 5}, boundary={0: 'nearest', 1: 'nearest'})
    tarr = trim_internal(garr, {0: 5, 1: 5}, boundary='nearest')
    assert_array_almost_equal(tarr, a)
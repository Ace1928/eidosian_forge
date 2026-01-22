from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_multiarray_defaults():
    x = da.ones((10,), chunks=10)
    y = da.ones((1, 10), chunks=5)
    z = da.map_overlap(lambda x, y: x + y, x, y, boundary='none')
    assert_eq(z.shape, (1, 10))
    assert_eq(z.sum(), 20.0)
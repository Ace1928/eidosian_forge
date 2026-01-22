from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('boundary', ['reflect', 'periodic', 'nearest', 'none'])
def test_trim_boundary(boundary):
    x = da.from_array(np.arange(24).reshape(4, 6), chunks=(2, 3))
    x_overlaped = da.overlap.overlap(x, 2, boundary={0: 'reflect', 1: boundary})
    x_trimmed = da.overlap.trim_overlap(x_overlaped, 2, boundary={0: 'reflect', 1: boundary})
    assert np.all(x == x_trimmed)
    x_overlaped = da.overlap.overlap(x, 2, boundary={1: boundary})
    x_trimmed = da.overlap.trim_overlap(x_overlaped, 2, boundary={1: boundary})
    assert np.all(x == x_trimmed)
    x_overlaped = da.overlap.overlap(x, 2, boundary=boundary)
    x_trimmed = da.overlap.trim_overlap(x_overlaped, 2, boundary=boundary)
    assert np.all(x == x_trimmed)
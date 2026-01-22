from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('drop_axis', ((0,), (1,), (2,), (0, 1), (1, 2), (2, 0), 1, (-3,), (-2,), (-1,), (-3, -2), (-2, -1), (-1, -3), -2))
def test_map_overlap_trim_using_drop_axis_and_different_depths(drop_axis):
    x = da.random.default_rng().standard_normal((5, 10, 8), chunks=(2, 5, 4))

    def _mean(x):
        return x.mean(axis=drop_axis)
    expected = _mean(x)
    boundary = (0, 'reflect', 'nearest')
    depth = (1, 3, 2)
    _drop_axis = (drop_axis,) if np.isscalar(drop_axis) else drop_axis
    _drop_axis = [d % x.ndim for d in _drop_axis]
    depth = tuple((0 if i in _drop_axis else d for i, d in enumerate(depth)))
    y = da.map_overlap(_mean, x, depth=depth, boundary=boundary, drop_axis=drop_axis, dtype=float).compute()
    assert_array_almost_equal(expected, y)
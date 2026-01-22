from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_bottleneck
@pytest.mark.parametrize('y, lengths_expected', [[np.arange(9), [[1, 0, 7, 7, 7, 7, 7, 7, 0], [3, 3, 3, 0, 3, 3, 0, 2, 2]]], [np.arange(9) * 3, [[3, 0, 21, 21, 21, 21, 21, 21, 0], [9, 9, 9, 0, 9, 9, 0, 6, 6]]], [[0, 2, 5, 6, 7, 8, 10, 12, 14], [[2, 0, 12, 12, 12, 12, 12, 12, 0], [6, 6, 6, 0, 4, 4, 0, 4, 4]]]])
def test_interpolate_na_nan_block_lengths(y, lengths_expected):
    arr = [[np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4], [np.nan, np.nan, np.nan, 1, np.nan, np.nan, 4, np.nan, np.nan]]
    da = xr.DataArray(arr, dims=['x', 'y'], coords={'x': [0, 1], 'y': y})
    index = get_clean_interp_index(da, dim='y', use_coordinate=True)
    actual = _get_nan_block_lengths(da, dim='y', index=index)
    expected = da.copy(data=lengths_expected)
    assert_equal(actual, expected)
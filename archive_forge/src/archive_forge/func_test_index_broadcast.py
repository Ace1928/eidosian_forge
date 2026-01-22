from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_index_broadcast(dtype) -> None:
    values = xr.DataArray(['ABCDEFGEFDBCA', 'BCDEFEFEFDBC', 'DEFBCGHIEFBC', 'EFGHBCEFBCBCBCEF'], dims=['X'])
    values = values.astype(dtype)
    sub = xr.DataArray(['EF', 'BC'], dims=['Y']).astype(dtype)
    start = xr.DataArray([0, 6], dims=['Z'])
    end = xr.DataArray([6, 12], dims=['Z'])
    result_0 = values.str.index(sub, start, end)
    result_1 = values.str.index(sub, start, end, side='left')
    expected = xr.DataArray([[[4, 7], [1, 10]], [[3, 7], [0, 10]], [[1, 8], [3, 10]], [[0, 6], [4, 8]]], dims=['X', 'Y', 'Z'])
    assert result_0.dtype == expected.dtype
    assert result_1.dtype == expected.dtype
    assert_equal(result_0, expected)
    assert_equal(result_1, expected)
    result_0 = values.str.rindex(sub, start, end)
    result_1 = values.str.index(sub, start, end, side='right')
    expected = xr.DataArray([[[4, 7], [1, 10]], [[3, 7], [0, 10]], [[1, 8], [3, 10]], [[0, 6], [4, 10]]], dims=['X', 'Y', 'Z'])
    assert result_0.dtype == expected.dtype
    assert result_1.dtype == expected.dtype
    assert_equal(result_0, expected)
    assert_equal(result_1, expected)
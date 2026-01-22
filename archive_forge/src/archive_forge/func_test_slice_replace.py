from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_slice_replace(dtype) -> None:
    da = lambda x: xr.DataArray(x).astype(dtype)
    values = da(['short', 'a bit longer', 'evenlongerthanthat', ''])
    expected = da(['shrt', 'a it longer', 'evnlongerthanthat', ''])
    result = values.str.slice_replace(2, 3)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['shzrt', 'a zit longer', 'evznlongerthanthat', 'z'])
    result = values.str.slice_replace(2, 3, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['shzort', 'a zbit longer', 'evzenlongerthanthat', 'z'])
    result = values.str.slice_replace(2, 2, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['shzort', 'a zbit longer', 'evzenlongerthanthat', 'z'])
    result = values.str.slice_replace(2, 1, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['shorz', 'a bit longez', 'evenlongerthanthaz', 'z'])
    result = values.str.slice_replace(-1, None, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['zrt', 'zer', 'zat', 'z'])
    result = values.str.slice_replace(None, -2, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['shortz', 'a bit znger', 'evenlozerthanthat', 'z'])
    result = values.str.slice_replace(6, 8, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = da(['zrt', 'a zit longer', 'evenlongzerthanthat', 'z'])
    result = values.str.slice_replace(-10, 3, 'z')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
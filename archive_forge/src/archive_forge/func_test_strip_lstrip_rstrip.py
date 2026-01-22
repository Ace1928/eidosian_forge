from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_strip_lstrip_rstrip(dtype) -> None:
    values = xr.DataArray(['  aa   ', ' bb \n', 'cc  ']).astype(dtype)
    result = values.str.strip()
    expected = xr.DataArray(['aa', 'bb', 'cc']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.lstrip()
    expected = xr.DataArray(['aa   ', 'bb \n', 'cc  ']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.rstrip()
    expected = xr.DataArray(['  aa', ' bb', 'cc']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
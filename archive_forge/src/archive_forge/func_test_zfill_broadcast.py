from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_zfill_broadcast(dtype) -> None:
    values = xr.DataArray(['1', '22', 'aaa', '333', '45678']).astype(dtype)
    width = np.array([4, 5, 0, 3, 8])
    result = values.str.zfill(width)
    expected = xr.DataArray(['0001', '00022', 'aaa', '333', '00045678']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
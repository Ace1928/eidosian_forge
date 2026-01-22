from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_count_broadcast(dtype) -> None:
    values = xr.DataArray(['foo', 'foofoo', 'foooofooofommmfoo']).astype(dtype)
    pat_str = np.array(['f[o]+', 'o', 'm']).astype(dtype)
    pat_re = np.array([re.compile(x) for x in pat_str])
    result_str = values.str.count(pat_str)
    result_re = values.str.count(pat_re)
    expected = xr.DataArray([1, 4, 3])
    assert result_str.dtype == expected.dtype
    assert result_re.dtype == expected.dtype
    assert_equal(result_str, expected)
    assert_equal(result_re, expected)
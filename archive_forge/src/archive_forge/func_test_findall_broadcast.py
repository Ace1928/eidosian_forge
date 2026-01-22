from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_findall_broadcast(dtype) -> None:
    value = xr.DataArray(['a_Xy_0', 'ab_xY_10', 'abc_Xy_01'], dims=['X']).astype(dtype)
    pat_str = xr.DataArray(['(\\w+)_Xy_\\d*', '\\w+_Xy_(\\d*)'], dims=['Y']).astype(dtype)
    pat_re = value.str._re_compile(pat=pat_str)
    expected_list: list[list[list]] = [[['a'], ['0']], [[], []], [['abc'], ['01']]]
    expected_dtype = [[[dtype(x) for x in y] for y in z] for z in expected_list]
    expected_np = np.array(expected_dtype, dtype=np.object_)
    expected = xr.DataArray(expected_np, dims=['X', 'Y'])
    res_str = value.str.findall(pat=pat_str)
    res_re = value.str.findall(pat=pat_re)
    assert res_str.dtype == expected.dtype
    assert res_re.dtype == expected.dtype
    assert_equal(res_str, expected)
    assert_equal(res_re, expected)
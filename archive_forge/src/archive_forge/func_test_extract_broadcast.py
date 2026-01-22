from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extract_broadcast(dtype) -> None:
    value = xr.DataArray(['a_Xy_0', 'ab_xY_10', 'abc_Xy_01'], dims=['X']).astype(dtype)
    pat_str = xr.DataArray(['(\\w+)_Xy_(\\d*)', '(\\w+)_xY_(\\d*)'], dims=['Y']).astype(dtype)
    pat_compiled = value.str._re_compile(pat=pat_str)
    expected_list = [[['a', '0'], ['', '']], [['', ''], ['ab', '10']], [['abc', '01'], ['', '']]]
    expected = xr.DataArray(expected_list, dims=['X', 'Y', 'Zz']).astype(dtype)
    res_str = value.str.extract(pat=pat_str, dim='Zz')
    res_re = value.str.extract(pat=pat_compiled, dim='Zz')
    assert res_str.dtype == expected.dtype
    assert res_re.dtype == expected.dtype
    assert_equal(res_str, expected)
    assert_equal(res_re, expected)
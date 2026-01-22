from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extract_multi_case(dtype) -> None:
    pat_str = '(\\w+)_Xy_(\\d*)'
    pat_re: str | bytes = pat_str if dtype == np.str_ else bytes(pat_str, encoding='UTF-8')
    pat_compiled = re.compile(pat_re)
    value = xr.DataArray([['a_Xy_0', 'ab_xY_10-bab_Xy_110-baab_Xy_1100', 'abc_Xy_01-cbc_Xy_2210'], ['abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210', '', 'abcdef_Xy_101-fef_Xy_5543210']], dims=['X', 'Y']).astype(dtype)
    expected = xr.DataArray([[['a', '0'], ['bab', '110'], ['abc', '01']], [['abcd', ''], ['', ''], ['abcdef', '101']]], dims=['X', 'Y', 'XX']).astype(dtype)
    res_str = value.str.extract(pat=pat_str, dim='XX')
    res_re = value.str.extract(pat=pat_compiled, dim='XX')
    res_str_case = value.str.extract(pat=pat_str, dim='XX', case=True)
    assert res_str.dtype == expected.dtype
    assert res_re.dtype == expected.dtype
    assert res_str_case.dtype == expected.dtype
    assert_equal(res_str, expected)
    assert_equal(res_re, expected)
    assert_equal(res_str_case, expected)
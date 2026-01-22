from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extract_single_nocase(dtype) -> None:
    pat_str = '(\\w+)?_Xy_\\d*'
    pat_re: str | bytes = pat_str if dtype == np.str_ else bytes(pat_str, encoding='UTF-8')
    pat_compiled = re.compile(pat_re, flags=re.IGNORECASE)
    value = xr.DataArray([['a_Xy_0', 'ab_xY_10-bab_Xy_110-baab_Xy_1100', 'abc_Xy_01-cbc_Xy_2210'], ['abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210', '_Xy_1', 'abcdef_Xy_101-fef_Xy_5543210']], dims=['X', 'Y']).astype(dtype)
    targ_none = xr.DataArray([['a', 'ab', 'abc'], ['abcd', '', 'abcdef']], dims=['X', 'Y']).astype(dtype)
    targ_dim = xr.DataArray([[['a'], ['ab'], ['abc']], [['abcd'], [''], ['abcdef']]], dims=['X', 'Y', 'XX']).astype(dtype)
    res_str_none = value.str.extract(pat=pat_str, dim=None, case=False)
    res_str_dim = value.str.extract(pat=pat_str, dim='XX', case=False)
    res_re_none = value.str.extract(pat=pat_compiled, dim=None)
    res_re_dim = value.str.extract(pat=pat_compiled, dim='XX')
    assert res_re_dim.dtype == targ_none.dtype
    assert res_str_dim.dtype == targ_dim.dtype
    assert res_re_none.dtype == targ_none.dtype
    assert res_re_dim.dtype == targ_dim.dtype
    assert_equal(res_str_none, targ_none)
    assert_equal(res_str_dim, targ_dim)
    assert_equal(res_re_none, targ_none)
    assert_equal(res_re_dim, targ_dim)
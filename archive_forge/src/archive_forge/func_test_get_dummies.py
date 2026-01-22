from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_get_dummies(dtype) -> None:
    values_line = xr.DataArray([['a|ab~abc|abc', 'ab', 'a||abc|abcd'], ['abcd|ab|a', 'abc|ab~abc', '|a']], dims=['X', 'Y']).astype(dtype)
    values_comma = xr.DataArray([['a~ab|abc~~abc', 'ab', 'a~abc~abcd'], ['abcd~ab~a', 'abc~ab|abc', '~a']], dims=['X', 'Y']).astype(dtype)
    vals_line = np.array(['a', 'ab', 'abc', 'abcd', 'ab~abc']).astype(dtype)
    vals_comma = np.array(['a', 'ab', 'abc', 'abcd', 'ab|abc']).astype(dtype)
    expected_list = [[[True, False, True, False, True], [False, True, False, False, False], [True, False, True, True, False]], [[True, True, False, True, False], [False, False, True, False, True], [True, False, False, False, False]]]
    expected_np = np.array(expected_list)
    expected = xr.DataArray(expected_np, dims=['X', 'Y', 'ZZ'])
    targ_line = expected.copy()
    targ_comma = expected.copy()
    targ_line.coords['ZZ'] = vals_line
    targ_comma.coords['ZZ'] = vals_comma
    res_default = values_line.str.get_dummies(dim='ZZ')
    res_line = values_line.str.get_dummies(dim='ZZ', sep='|')
    res_comma = values_comma.str.get_dummies(dim='ZZ', sep='~')
    assert res_default.dtype == targ_line.dtype
    assert res_line.dtype == targ_line.dtype
    assert res_comma.dtype == targ_comma.dtype
    assert_equal(res_default, targ_line)
    assert_equal(res_line, targ_line)
    assert_equal(res_comma, targ_comma)
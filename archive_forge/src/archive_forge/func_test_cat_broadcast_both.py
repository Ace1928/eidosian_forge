from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_cat_broadcast_both(dtype) -> None:
    values_1 = xr.DataArray(['a', 'bb', 'cccc'], dims=['Y']).astype(dtype)
    values_2 = xr.DataArray(['11111', '4'], dims=['X'])
    targ_blank = xr.DataArray([['a11111', 'bb11111', 'cccc11111'], ['a4', 'bb4', 'cccc4']], dims=['X', 'Y']).astype(dtype).T
    targ_space = xr.DataArray([['a 11111', 'bb 11111', 'cccc 11111'], ['a 4', 'bb 4', 'cccc 4']], dims=['X', 'Y']).astype(dtype).T
    targ_bars = xr.DataArray([['a||11111', 'bb||11111', 'cccc||11111'], ['a||4', 'bb||4', 'cccc||4']], dims=['X', 'Y']).astype(dtype).T
    targ_comma = xr.DataArray([['a, 11111', 'bb, 11111', 'cccc, 11111'], ['a, 4', 'bb, 4', 'cccc, 4']], dims=['X', 'Y']).astype(dtype).T
    res_blank = values_1.str.cat(values_2)
    res_add = values_1.str + values_2
    res_space = values_1.str.cat(values_2, sep=' ')
    res_bars = values_1.str.cat(values_2, sep='||')
    res_comma = values_1.str.cat(values_2, sep=', ')
    assert res_blank.dtype == targ_blank.dtype
    assert res_add.dtype == targ_blank.dtype
    assert res_space.dtype == targ_space.dtype
    assert res_bars.dtype == targ_bars.dtype
    assert res_comma.dtype == targ_comma.dtype
    assert_equal(res_blank, targ_blank)
    assert_equal(res_add, targ_blank)
    assert_equal(res_space, targ_space)
    assert_equal(res_bars, targ_bars)
    assert_equal(res_comma, targ_comma)
from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize(['func', 'expected'], [pytest.param(lambda x: x.str.isalnum(), [True, True, True, True, True, False, True, True, False, False], id='isalnum'), pytest.param(lambda x: x.str.isalpha(), [True, True, True, False, False, False, True, False, False, False], id='isalpha'), pytest.param(lambda x: x.str.isdigit(), [False, False, False, True, False, False, False, True, False, False], id='isdigit'), pytest.param(lambda x: x.str.islower(), [False, True, False, False, False, False, False, False, False, False], id='islower'), pytest.param(lambda x: x.str.isspace(), [False, False, False, False, False, False, False, False, False, True], id='isspace'), pytest.param(lambda x: x.str.istitle(), [True, False, True, False, True, False, False, False, False, False], id='istitle'), pytest.param(lambda x: x.str.isupper(), [True, False, False, False, True, False, True, False, False, False], id='isupper')])
def test_ismethods(dtype, func: Callable[[xr.DataArray], xr.DataArray], expected: list[bool]) -> None:
    values = xr.DataArray(['A', 'b', 'Xy', '4', '3A', '', 'TT', '55', '-', '  ']).astype(dtype)
    expected_da = xr.DataArray(expected)
    actual = func(values)
    assert actual.dtype == expected_da.dtype
    assert_equal(actual, expected_da)
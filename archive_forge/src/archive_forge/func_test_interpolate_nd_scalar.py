from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('method', ['linear'])
@pytest.mark.parametrize('case', [pytest.param(3, id='no_chunk'), pytest.param(4, id='chunked')])
def test_interpolate_nd_scalar(method: InterpOptions, case: int) -> None:
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    if not has_dask and case in [4]:
        pytest.skip('dask is not installed in the environment.')
    da = get_example_data(case)
    xdest = 0.4
    ydest = 0.05
    actual = da.interp(x=xdest, y=ydest, method=method)
    expected_data = scipy.interpolate.RegularGridInterpolator((da['x'], da['y']), da.transpose('x', 'y', 'z').values, method='linear', bounds_error=False, fill_value=np.nan)(np.stack([xdest, ydest], axis=-1))
    coords = {'x': xdest, 'y': ydest, 'x2': da['x2'].interp(x=xdest), 'z': da['z']}
    expected = xr.DataArray(expected_data[0], dims=['z'], coords=coords)
    assert_allclose(actual, expected)
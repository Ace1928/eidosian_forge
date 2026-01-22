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
@requires_scipy
def test_interpolate_nd_nd() -> None:
    """Interpolate nd array with an nd indexer sharing coordinates."""
    a = [0, 2]
    x = [0, 1, 2]
    da = xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'x'), coords={'a': a, 'x': x})
    y = [10]
    c = {'x': x, 'y': y}
    ia = xr.DataArray([[1, 2, 2]], dims=('y', 'x'), coords=c)
    out = da.interp(a=ia)
    expected = xr.DataArray([[1.5, 4, 5]], dims=('y', 'x'), coords=c)
    xr.testing.assert_allclose(out.drop_vars('a'), expected)
    with pytest.raises(ValueError):
        c = {'x': [1], 'y': y}
        ia = xr.DataArray([[1]], dims=('y', 'x'), coords=c)
        da.interp(a=ia)
    with pytest.raises(ValueError):
        c = {'x': [5, 6, 7], 'y': y}
        ia = xr.DataArray([[1]], dims=('y', 'x'), coords=c)
        da.interp(a=ia)
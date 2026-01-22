from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_cftime
@requires_dask
@pytest.mark.parametrize('use_dask', [False, True])
def test_cftime_ceil_accessor(cftime_rounding_dataarray, cftime_date_type, use_dask) -> None:
    import dask.array as da
    freq = 'D'
    expected = xr.DataArray([[cftime_date_type(1, 1, 2, 0), cftime_date_type(1, 1, 2, 0)], [cftime_date_type(1, 1, 2, 0), cftime_date_type(1, 1, 3, 0)]], name='ceil')
    if use_dask:
        chunks = {'dim_0': 1}
        with raise_if_dask_computes(max_computes=1):
            result = cftime_rounding_dataarray.chunk(chunks).dt.ceil(freq)
        expected = expected.chunk(chunks)
        assert isinstance(result.data, da.Array)
        assert result.chunks == expected.chunks
    else:
        result = cftime_rounding_dataarray.dt.ceil(freq)
    assert_identical(result, expected)
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_dask
@pytest.mark.parametrize('field', ['days', 'seconds', 'microseconds', 'nanoseconds'])
def test_dask_field_access(self, field) -> None:
    import dask.array as da
    expected = getattr(self.times_data.dt, field)
    dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
    dask_times_2d = xr.DataArray(dask_times_arr, coords=self.data.coords, dims=self.data.dims, name='data')
    with raise_if_dask_computes():
        actual = getattr(dask_times_2d.dt, field)
    assert isinstance(actual.data, da.Array)
    assert_chunks_equal(actual, dask_times_2d)
    assert_equal(actual, expected)
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_dask
@pytest.mark.parametrize('method, parameters', [('floor', 'D'), ('ceil', 'D'), ('round', 'D')])
def test_dask_accessor_method(self, method, parameters) -> None:
    import dask.array as da
    expected = getattr(self.times_data.dt, method)(parameters)
    dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
    dask_times_2d = xr.DataArray(dask_times_arr, coords=self.data.coords, dims=self.data.dims, name='data')
    with raise_if_dask_computes():
        actual = getattr(dask_times_2d.dt, method)(parameters)
    assert isinstance(actual.data, da.Array)
    assert_chunks_equal(actual, dask_times_2d)
    assert_equal(actual.compute(), expected.compute())
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@requires_cftime
@requires_dask
@pytest.mark.parametrize('time_chunks', (1, 5))
@pytest.mark.parametrize('resample_spec', ('1YS', '5YS', '10YS'))
def test_weighted_lazy_resample(time_chunks, resample_spec):

    def mean_func(ds):
        return ds.weighted(ds.weights).mean('time')
    t = xr.cftime_range(start='2000', periods=20, freq='1YS')
    weights = xr.DataArray(np.random.rand(len(t)), dims=['time'], coords={'time': t})
    data = xr.DataArray(np.random.rand(len(t)), dims=['time'], coords={'time': t, 'weights': weights})
    ds = xr.Dataset({'data': data}).chunk({'time': time_chunks})
    with raise_if_dask_computes():
        ds.resample(time=resample_spec).map(mean_func)
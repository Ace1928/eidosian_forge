from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
def test_decode_cf_time_bounds() -> None:
    da = DataArray(np.arange(6, dtype='int64').reshape((3, 2)), coords={'time': [1, 2, 3]}, dims=('time', 'nbnd'), name='time_bnds')
    attrs = {'units': 'days since 2001-01', 'calendar': 'standard', 'bounds': 'time_bnds'}
    ds = da.to_dataset()
    ds['time'].attrs.update(attrs)
    _update_bounds_attributes(ds.variables)
    assert ds.variables['time_bnds'].attrs == {'units': 'days since 2001-01', 'calendar': 'standard'}
    dsc = decode_cf(ds)
    assert dsc.time_bnds.dtype == np.dtype('M8[ns]')
    dsc = decode_cf(ds, decode_times=False)
    assert dsc.time_bnds.dtype == np.dtype('int64')
    ds = da.to_dataset()
    ds['time'].attrs.update(attrs)
    bnd_attr = {'units': 'hours since 2001-01', 'calendar': 'noleap'}
    ds['time_bnds'].attrs.update(bnd_attr)
    _update_bounds_attributes(ds.variables)
    assert ds.variables['time_bnds'].attrs == bnd_attr
    ds = da.to_dataset()
    ds['time'].attrs.update(attrs)
    ds['time'].attrs['bounds'] = 'fake_var'
    _update_bounds_attributes(ds.variables)
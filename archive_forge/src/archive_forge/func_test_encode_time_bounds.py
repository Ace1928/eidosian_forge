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
@requires_cftime
def test_encode_time_bounds() -> None:
    time = pd.date_range('2000-01-16', periods=1)
    time_bounds = pd.date_range('2000-01-01', periods=2, freq='MS')
    ds = Dataset(dict(time=time, time_bounds=time_bounds))
    ds.time.attrs = {'bounds': 'time_bounds'}
    ds.time.encoding = {'calendar': 'noleap', 'units': 'days since 2000-01-01'}
    expected = {}
    expected['time_bounds'] = Variable(data=np.array([0, 31]), dims=['time_bounds'])
    encoded, _ = cf_encoder(ds.variables, ds.attrs)
    assert_equal(encoded['time_bounds'], expected['time_bounds'])
    assert 'calendar' not in encoded['time_bounds'].attrs
    assert 'units' not in encoded['time_bounds'].attrs
    ds.time_bounds.encoding = {'calendar': 'noleap', 'units': 'days since 2000-01-01'}
    encoded, _ = cf_encoder({k: v for k, v in ds.variables.items()}, ds.attrs)
    assert_equal(encoded['time_bounds'], expected['time_bounds'])
    assert 'calendar' not in encoded['time_bounds'].attrs
    assert 'units' not in encoded['time_bounds'].attrs
    ds.time_bounds.encoding = {'calendar': 'noleap', 'units': 'days since 1849-01-01'}
    encoded, _ = cf_encoder({k: v for k, v in ds.variables.items()}, ds.attrs)
    with pytest.raises(AssertionError):
        assert_equal(encoded['time_bounds'], expected['time_bounds'])
    assert 'calendar' not in encoded['time_bounds'].attrs
    assert encoded['time_bounds'].attrs['units'] == ds.time_bounds.encoding['units']
    ds.time.encoding = {}
    with pytest.warns(UserWarning):
        cf_encoder(ds.variables, ds.attrs)
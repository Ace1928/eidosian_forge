from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@arm_xfail
@pytest.mark.filterwarnings('ignore:All-NaN .* encountered:RuntimeWarning')
@pytest.mark.parametrize('dask', [False, True] if has_dask else [False])
def test_datetime_mean(dask: bool) -> None:
    da = DataArray(np.array(['2010-01-01', 'NaT', '2010-01-03', 'NaT', 'NaT'], dtype='M8[ns]'), dims=['time'])
    if dask:
        da = da.chunk({'time': 3})
    expect = DataArray(np.array('2010-01-02', dtype='M8[ns]'))
    expect_nat = DataArray(np.array('NaT', dtype='M8[ns]'))
    actual = da.mean()
    if dask:
        assert actual.chunks is not None
    assert_equal(actual, expect)
    actual = da.mean(skipna=False)
    if dask:
        assert actual.chunks is not None
    assert_equal(actual, expect_nat)
    assert_equal(da[[1]].mean(), expect_nat)
    assert_equal(da[[1]].mean(skipna=False), expect_nat)
    assert_equal(da[0].mean(), da[0])
    assert_equal(da[0].mean(skipna=False), da[0])
    assert_equal(da[1].mean(), expect_nat)
    assert_equal(da[1].mean(skipna=False), expect_nat)
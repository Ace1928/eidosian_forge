from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_equally_weighted_cov_corr() -> None:
    da = xr.DataArray(np.random.random((3, 21, 4)), coords={'time': pd.date_range('2000-01-01', freq='1D', periods=21)}, dims=('a', 'time', 'x'))
    db = xr.DataArray(np.random.random((3, 21, 4)), coords={'time': pd.date_range('2000-01-01', freq='1D', periods=21)}, dims=('a', 'time', 'x'))
    assert_allclose(xr.cov(da, db, weights=None), xr.cov(da, db, weights=xr.DataArray(1)))
    assert_allclose(xr.cov(da, db, weights=None), xr.cov(da, db, weights=xr.DataArray(2)))
    assert_allclose(xr.corr(da, db, weights=None), xr.corr(da, db, weights=xr.DataArray(1)))
    assert_allclose(xr.corr(da, db, weights=None), xr.corr(da, db, weights=xr.DataArray(2)))
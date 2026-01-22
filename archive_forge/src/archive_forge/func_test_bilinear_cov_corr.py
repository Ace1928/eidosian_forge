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
@pytest.mark.parametrize('weighted', [True, False])
def test_bilinear_cov_corr(weighted: bool) -> None:
    da = xr.DataArray(np.random.random((3, 21, 4)), coords={'time': pd.date_range('2000-01-01', freq='1D', periods=21)}, dims=('a', 'time', 'x'))
    db = xr.DataArray(np.random.random((3, 21, 4)), coords={'time': pd.date_range('2000-01-01', freq='1D', periods=21)}, dims=('a', 'time', 'x'))
    dc = xr.DataArray(np.random.random((3, 21, 4)), coords={'time': pd.date_range('2000-01-01', freq='1D', periods=21)}, dims=('a', 'time', 'x'))
    if weighted:
        weights = xr.DataArray(np.abs(np.random.random(4)), dims='x')
    else:
        weights = None
    k = np.random.random(1)[0]
    assert_allclose(xr.cov(da + k, db, weights=weights), xr.cov(da, db, weights=weights))
    assert_allclose(xr.cov(da, db + k, weights=weights), xr.cov(da, db, weights=weights))
    assert_allclose(xr.cov(da + dc, db, weights=weights), xr.cov(da, db, weights=weights) + xr.cov(dc, db, weights=weights))
    assert_allclose(xr.cov(da, db + dc, weights=weights), xr.cov(da, db, weights=weights) + xr.cov(da, dc, weights=weights))
    assert_allclose(xr.cov(k * da, db, weights=weights), k * xr.cov(da, db, weights=weights))
    assert_allclose(xr.cov(da, k * db, weights=weights), k * xr.cov(da, db, weights=weights))
    assert_allclose(xr.corr(da + k, db, weights=weights), xr.corr(da, db, weights=weights))
    assert_allclose(xr.corr(da, db + k, weights=weights), xr.corr(da, db, weights=weights))
    assert_allclose(xr.corr(k * da, db, weights=weights), xr.corr(da, db, weights=weights))
    assert_allclose(xr.corr(da, k * db, weights=weights), xr.corr(da, db, weights=weights))
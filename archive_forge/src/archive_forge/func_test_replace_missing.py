from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
@pytest.mark.skipif(PLATFORM_WIN32, reason='Windows 32-bit')
def test_replace_missing(self):
    x = self.x.copy()
    x[::5, ::7] = np.nan
    pc = PCA(x, missing='drop-row')
    x_dropped_row = x[np.logical_not(np.any(np.isnan(x), 1))]
    pc_dropped = PCA(x_dropped_row)
    assert_allclose(pc.projection, pc_dropped.projection)
    assert_equal(x, pc.data)
    pc = PCA(x, missing='drop-col')
    x_dropped_col = x[:, np.logical_not(np.any(np.isnan(x), 0))]
    pc_dropped = PCA(x_dropped_col)
    assert_allclose(pc.projection, pc_dropped.projection)
    assert_equal(x, pc.data)
    pc = PCA(x, missing='drop-min')
    if x_dropped_row.size > x_dropped_col.size:
        x_dropped_min = x_dropped_row
    else:
        x_dropped_min = x_dropped_col
    pc_dropped = PCA(x_dropped_min)
    assert_allclose(pc.projection, pc_dropped.projection)
    assert_equal(x, pc.data)
    pc = PCA(x, ncomp=3, missing='fill-em')
    missing = np.isnan(x)
    mu = np.nanmean(x, axis=0)
    errors = x - mu
    sigma = np.sqrt(np.nanmean(errors ** 2, axis=0))
    x_std = errors / sigma
    x_std[missing] = 0.0
    last = x_std[missing]
    delta = 1.0
    count = 0
    while delta > 5e-08:
        pc_temp = PCA(x_std, ncomp=3, standardize=False, demean=False)
        x_std[missing] = pc_temp.projection[missing]
        current = x_std[missing]
        diff = current - last
        delta = np.sqrt(np.sum(diff ** 2)) / np.sqrt(np.sum(current ** 2))
        last = current
        count += 1
    x = self.x + 0.0
    projection = pc_temp.projection * sigma + mu
    x[missing] = projection[missing]
    assert_allclose(pc._adjusted_data, x)
    assert_equal(self.x, self.x_copy)
    x = self.x
    pc = PCA(x)
    pc_dropped = PCA(x, missing='drop-row')
    assert_allclose(pc.projection, pc_dropped.projection, atol=DECIMAL_5)
    pc_dropped = PCA(x, missing='drop-col')
    assert_allclose(pc.projection, pc_dropped.projection, atol=DECIMAL_5)
    pc_dropped = PCA(x, missing='drop-min')
    assert_allclose(pc.projection, pc_dropped.projection, atol=DECIMAL_5)
    pc = PCA(x, ncomp=3)
    pc_dropped = PCA(x, ncomp=3, missing='fill-em')
    assert_allclose(pc.projection, pc_dropped.projection, atol=DECIMAL_5)
    x = self.x.copy()
    x[:, :] = np.nan
    assert_raises(ValueError, PCA, x, missing='drop-row')
    assert_raises(ValueError, PCA, x, missing='drop-col')
    assert_raises(ValueError, PCA, x, missing='drop-min')
    assert_raises(ValueError, PCA, x, missing='fill-em')
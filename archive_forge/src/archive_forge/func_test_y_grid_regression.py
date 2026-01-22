import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_y_grid_regression():
    y_grid = arange(1000)
    _, pv = etest_poisson_2indep(60, 51477.5, 30, 54308.7, y_grid=y_grid)
    assert_allclose(pv, 0.000567261758250953, atol=1e-15)
    _, pv = etest_poisson_2indep(41, 28010, 15, 19017, y_grid=y_grid)
    assert_allclose(pv, 0.03782053187021494, atol=1e-15)
    _, pv = etest_poisson_2indep(1, 1, 1, 1, y_grid=[1])
    assert_allclose(pv, 0.1353352832366127, atol=1e-15)
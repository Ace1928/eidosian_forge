import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@pytest.mark.parametrize('method', ['nearest', 'linear'])
def test_nan_x_2d(self, method):
    x, y = (np.array([0, 1, 2]), np.array([1, 3, 7]))

    def f(x, y):
        return x ** 2 + y ** 2
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
    data = f(xg, yg)
    interp = RegularGridInterpolator((x, y), data, method=method, bounds_error=False)
    with np.errstate(invalid='ignore'):
        res = interp([[1.5, np.nan], [1, 1]])
    assert_allclose(res[1], 2, atol=1e-14)
    assert np.isnan(res[0])
    rng = np.random.default_rng(8143215468)
    x = rng.random(size=100) * 4 - 1
    y = rng.random(size=100) * 8
    i1 = rng.random(size=100) > 0.5
    i2 = rng.random(size=100) > 0.5
    i = i1 | i2
    x[i1] = np.nan
    y[i2] = np.nan
    z = np.array([x, y]).T
    with np.errstate(invalid='ignore'):
        res = interp(z)
    assert_equal(res[i], np.nan)
    assert_equal(res[~i], interp(z[~i]))
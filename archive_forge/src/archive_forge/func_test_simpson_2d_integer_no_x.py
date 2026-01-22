import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
@pytest.mark.parametrize('droplast', [False, True])
def test_simpson_2d_integer_no_x(self, droplast):
    y = np.array([[2, 2, 4, 4, 8, 8, -4, 5], [4, 4, 2, -4, 10, 22, -2, 10]])
    if droplast:
        y = y[:, :-1]
    result = simpson(y, axis=-1)
    expected = simpson(np.array(y, dtype=np.float64), axis=-1)
    assert_equal(result, expected)
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
@pytest.mark.parametrize('initial', [1, 0.5])
def test_initial_warning(self, initial):
    """If initial is not None or 0, a ValueError is raised."""
    y = np.linspace(0, 10, num=10)
    with pytest.deprecated_call(match='`initial`'):
        res = cumulative_trapezoid(y, initial=initial)
    assert_allclose(res, [initial, *np.cumsum(y[1:] + y[:-1]) / 2])
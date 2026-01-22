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
@pytest.mark.parametrize('log', [False, True])
def test_zero(self, log):
    message = 'A lower limit was equal to an upper limit, so'
    with pytest.warns(UserWarning, match=message):
        res = qmc_quad(lambda x: 1, [0, 0], [0, 1], log=log)
    assert res.integral == (-np.inf if log else 0)
    assert res.standard_error == 0
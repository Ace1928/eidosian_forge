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
@given(y=hyp_num.arrays(np.float64, hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10), elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-07)))
def test_cumulative_simpson_against_simpson_with_default_dx(self, y):
    """Theoretically, the output of `cumulative_simpson` will be identical
        to `simpson` at all even indices and in the last index. The first index
        will not match as `simpson` uses the trapezoidal rule when there are only two
        data points. Odd indices after the first index are shown to match with
        a mathematically-derived correction."""

    def simpson_reference(y):
        return np.stack([simpson(y[..., :i], dx=1.0) for i in range(2, y.shape[-1] + 1)], axis=-1)
    res = cumulative_simpson(y, dx=1.0)
    ref = simpson_reference(y)
    theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(y, x=np.arange(y.shape[-1]))
    np.testing.assert_allclose(res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:])
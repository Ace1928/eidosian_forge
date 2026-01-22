import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
def test_tukeyhsd():
    res = '    pair      diff        lwr        upr       p adj\n    P-M   8.150000 -10.037586 26.3375861 0.670063958\n    S-M  -3.258333 -21.445919 14.9292527 0.982419709\n    T-M  23.808333   5.620747 41.9959194 0.006783701\n    V-M   4.791667 -13.395919 22.9792527 0.931020848\n    S-P -11.408333 -29.595919  6.7792527 0.360680099\n    T-P  15.658333  -2.529253 33.8459194 0.113221634\n    V-P  -3.358333 -21.545919 14.8292527 0.980350080\n    T-S  27.066667   8.879081 45.2542527 0.002027122\n    V-S   8.050000 -10.137586 26.2375861 0.679824487\n    V-T -19.016667 -37.204253 -0.8290806 0.037710044\n    '
    res = np.array([[8.15, -10.037586, 26.3375861, 0.670063958], [-3.258333, -21.445919, 14.9292527, 0.982419709], [23.808333, 5.620747, 41.9959194, 0.006783701], [4.791667, -13.395919, 22.9792527, 0.931020848], [-11.408333, -29.595919, 6.7792527, 0.360680099], [15.658333, -2.529253, 33.8459194, 0.113221634], [-3.358333, -21.545919, 14.8292527, 0.98035008], [27.066667, 8.879081, 45.2542527, 0.002027122], [8.05, -10.137586, 26.2375861, 0.679824487], [-19.016667, -37.204253, -0.8290806, 0.037710044]])
    m_r = [94.39167, 102.54167, 91.13333, 118.2, 99.18333]
    myres = tukeyhsd(m_r, 6, 110.8254416667, alpha=0.05, df=4)
    pairs, reject, meandiffs, std_pairs, confint, q_crit = myres[:6]
    assert_almost_equal(meandiffs, res[:, 0], decimal=5)
    assert_almost_equal(confint, res[:, 1:3], decimal=2)
    assert_equal(reject, res[:, 3] < 0.05)
    small_pvals_idx = [2, 5, 7, 9]
    scipy_version = version.parse(scipy.version.version) >= version.parse('1.7.0')
    rtol = 1e-05 if scipy_version else 0.01
    assert_allclose(myres[8][small_pvals_idx], res[small_pvals_idx, 3], rtol=rtol)
import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
@pytest.mark.parametrize('estimate_prob', [True, False])
@pytest.mark.parametrize('estimate_scale', [True, False])
@pytest.mark.parametrize('estimate_mean', [True, False])
def test_null_constrained(estimate_mean, estimate_scale, estimate_prob):
    grid = np.linspace(0.001, 0.999, 1000)
    z0 = norm.ppf(grid)
    z1 = np.linspace(3, 4, 20)
    zs = np.concatenate((z0, z1))
    emp_null = NullDistribution(zs, estimate_mean=estimate_mean, estimate_scale=estimate_scale, estimate_null_proportion=estimate_prob)
    if not estimate_mean:
        assert_allclose(emp_null.mean, 0, atol=1e-05, rtol=1e-05)
    if not estimate_scale:
        assert_allclose(emp_null.sd, 1, atol=1e-05, rtol=0.01)
    if not estimate_prob:
        assert_allclose(emp_null.null_proportion, 1, atol=1e-05, rtol=0.01)
    assert_allclose(emp_null.pdf(np.r_[-1, 0, 1]), norm.pdf(np.r_[-1, 0, 1], loc=emp_null.mean, scale=emp_null.sd), rtol=1e-13)
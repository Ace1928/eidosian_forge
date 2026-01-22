import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_mean_var(self):
    for m in [9, np.array([1, 5, 10])]:
        n, p = zinegbin.convert_params(m, 1, 1)
        nbinom_mean, nbinom_var = (nbinom.mean(n, p), nbinom.var(n, p))
        zinb_mean = zinegbin.mean(m, 1, 1, 0)
        zinb_var = zinegbin.var(m, 1, 1, 0)
        assert_allclose(nbinom_mean, zinb_mean, rtol=1e-10)
        assert_allclose(nbinom_var, zinb_var, rtol=1e-10)
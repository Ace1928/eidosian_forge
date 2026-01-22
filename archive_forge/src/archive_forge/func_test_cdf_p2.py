import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_cdf_p2(self):
    n, p = zinegbin.convert_params(30, 0.1, 2)
    nbinom_cdf = nbinom.cdf(10, n, p)
    zinbinom_cdf = zinegbin.cdf(10, 30, 0.1, 2, 0)
    assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)
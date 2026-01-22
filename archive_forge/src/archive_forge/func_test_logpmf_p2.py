import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_logpmf_p2(self):
    n, p = zinegbin.convert_params(10, 1, 2)
    nb_logpmf = nbinom.logpmf(200, n, p)
    tnb_logpmf = zinegbin.logpmf(200, 10, 1, 2, 0.01)
    assert_allclose(nb_logpmf, tnb_logpmf, rtol=0.01, atol=0.01)
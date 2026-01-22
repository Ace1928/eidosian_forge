import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_logpmf_zero(self):
    n, p = truncatednegbin.convert_params(5, 1, 2)
    nb_logpmf = nbinom.logpmf(1, n, p) - np.log(nbinom.sf(0, n, p))
    tnb_logpmf = truncatednegbin.logpmf(1, 5, 1, 2, 0)
    assert_allclose(nb_logpmf, tnb_logpmf, rtol=0.01, atol=0.01)
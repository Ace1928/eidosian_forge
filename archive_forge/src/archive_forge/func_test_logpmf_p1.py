import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
def test_logpmf_p1(self):
    poisson_pmf = poisson.logpmf(5, 2)
    genpoisson_pmf = genpoisson_p.logpmf(5, 2, 0, 1)
    assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)
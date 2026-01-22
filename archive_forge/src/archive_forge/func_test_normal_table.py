import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, \
def test_normal_table(self):
    np.random.seed(3975)
    x_n = stats.norm.rvs(size=500)
    d_ks_norm, p_norm = lilliefors(x_n, dist='norm', pvalmethod='table')
    assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
    assert_almost_equal(p_norm, 0.64175, decimal=3)
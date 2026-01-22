import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_confint_mean(self):
    d1w = self.d1w
    alpha = 0.05
    low, upp = d1w.tconfint_mean()
    t, p, d = d1w.ttest_mean(low)
    assert_almost_equal(p, alpha * np.ones(p.shape), 8)
    t, p, d = d1w.ttest_mean(upp)
    assert_almost_equal(p, alpha * np.ones(p.shape), 8)
    t, p, d = d1w.ttest_mean(np.vstack((low, upp)))
    assert_almost_equal(p, alpha * np.ones(p.shape), 8)
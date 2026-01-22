import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_ztest_ztost():
    import statsmodels.stats.proportion as smprop
    x1 = [0, 1]
    w1 = [5, 15]
    res2 = smprop.proportions_ztest(15, 20.0, value=0.5)
    d1 = DescrStatsW(x1, w1)
    res1 = d1.ztest_mean(0.5)
    assert_allclose(res1, res2, rtol=0.03, atol=0.003)
    d2 = DescrStatsW(x1, np.array(w1) * 21.0 / 20)
    res1 = d2.ztest_mean(0.5)
    assert_almost_equal(res1, res2, decimal=12)
    res1 = d2.ztost_mean(0.4, 0.6)
    res2 = smprop.proportions_ztost(15, 20.0, 0.4, 0.6)
    assert_almost_equal(res1[0], res2[0], decimal=12)
    x2 = [0, 1]
    w2 = [10, 10]
    d2 = DescrStatsW(x2, w2)
    res1 = ztest(d1.asrepeats(), d2.asrepeats())
    res2 = smprop.proportions_chisquare(np.asarray([15, 10]), np.asarray([20.0, 20]))
    assert_allclose(res1[1], res2[1], rtol=0.03)
    res1a = CompareMeans(d1, d2).ztest_ind()
    assert_allclose(res1a[1], res2[1], rtol=0.03)
    assert_almost_equal(res1a, res1, decimal=12)
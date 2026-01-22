from scipy import stats
from numpy.testing import assert_allclose
from statsmodels.stats.effect_size import (
def test_noncent_f():
    f_stat, df1, df2 = (3.5, 4, 75)
    ci_nc = [0.7781436, 29.72949219]
    res = _noncentrality_f(f_stat, df1, df2, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    mean = stats.ncf.mean(df1, df2, res.nc)
    assert_allclose(f_stat, mean, rtol=1e-08)
    assert_allclose(stats.ncf.cdf(f_stat, df1, df2, res.confint), [0.975, 0.025], rtol=5e-05)
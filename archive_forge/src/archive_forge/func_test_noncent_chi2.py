from scipy import stats
from numpy.testing import assert_allclose
from statsmodels.stats.effect_size import (
def test_noncent_chi2():
    chi2_stat, df = (7.5, 2)
    ci_nc = [0.03349255, 20.76049805]
    res = _noncentrality_chisquare(chi2_stat, df, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    mean = stats.ncx2.mean(df, res.nc)
    assert_allclose(chi2_stat, mean, rtol=1e-08)
    assert_allclose(stats.ncx2.cdf(chi2_stat, df, res.confint), [0.975, 0.025], rtol=1e-08)
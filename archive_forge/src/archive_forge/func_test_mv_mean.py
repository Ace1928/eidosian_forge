import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_mv_mean():
    x = np.asarray([[1.0, 24.0, 23.5, 1.0], [2.0, 25.0, 24.5, 1.0], [3.0, 21.0, 20.5, 1.0], [4.0, 22.0, 20.5, 1.0], [5.0, 23.0, 22.5, 1.0], [6.0, 18.0, 16.5, 1.0], [7.0, 17.0, 16.5, 1.0], [8.0, 28.0, 27.5, 1.0], [9.0, 24.0, 23.5, 1.0], [10.0, 27.0, 25.5, 1.0], [11.0, 21.0, 20.5, 1.0], [12.0, 23.0, 22.5, 1.0], [1.0, 20.0, 19.0, 0.0], [2.0, 23.0, 22.0, 0.0], [3.0, 21.0, 20.0, 0.0], [4.0, 25.0, 24.0, 0.0], [5.0, 18.0, 17.0, 0.0], [6.0, 17.0, 16.0, 0.0], [7.0, 18.0, 17.0, 0.0], [8.0, 24.0, 23.0, 0.0], [9.0, 20.0, 19.0, 0.0], [10.0, 24.0, 22.0, 0.0], [11.0, 23.0, 22.0, 0.0], [12.0, 19.0, 18.0, 0.0]])
    res = smmv.test_mvmean(x[:, 1:3], [21, 21])
    res_stata = Holder(p_F=1.25062334808e-09, df_r=22, df_m=2, F=59.91609589041116, T2=125.2791095890415)
    assert_allclose(res.statistic, res_stata.F, rtol=1e-10)
    assert_allclose(res.pvalue, res_stata.p_F, rtol=1e-10)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-10)
    assert_equal(res.df, [res_stata.df_m, res_stata.df_r])
    mask = x[:, -1] == 1
    x1 = x[mask, 1:3]
    x0 = x[~mask, 1:3]
    res_p = smmv.test_mvmean(x1 - x0, [0, 0])
    res_stata = Holder(T2=9.698067632850247, df=10, k=2, N=12, F=4.4082126, p_F=0.0424)
    res = res_p
    assert_allclose(res.statistic, res_stata.F, atol=5e-07)
    assert_allclose(res.pvalue, res_stata.p_F, atol=0.0005)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-10)
    assert_equal(res.df, [res_stata.k, res_stata.df])
    res_stata = Holder(p_F=0.0423949782937231, df_r=10, df_m=2, F=4.408212560386478, T2=9.69806763285025)
    assert_allclose(res.statistic, res_stata.F, rtol=1e-12)
    assert_allclose(res.pvalue, res_stata.p_F, rtol=1e-12)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-12)
    assert_equal(res.df, [res_stata.df_m, res_stata.df_r])
    dw = weightstats.DescrStatsW(x)
    ci0 = dw.tconfint_mean(alpha=0.05)
    nobs = len(x[:, 1:])
    ci1 = confint_mvmean_fromstats(dw.mean, np.diag(dw.var), nobs, lin_transf=np.eye(4), alpha=0.05)
    ci2 = confint_mvmean_fromstats(dw.mean, dw.cov, nobs, lin_transf=np.eye(4), alpha=0.05)
    assert_allclose(ci1[:2], ci0, rtol=1e-13)
    assert_allclose(ci2[:2], ci0, rtol=1e-13)
    res = smmv.confint_mvmean(x, lin_transf=np.eye(4), alpha=0.05)
    assert_allclose(res, ci2, rtol=1e-13)
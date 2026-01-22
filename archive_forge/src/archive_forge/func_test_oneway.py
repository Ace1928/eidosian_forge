import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pytest
from statsmodels.stats.robust_compare import (
import statsmodels.stats.oneway as smo
from statsmodels.tools.testing import Holder
from scipy.stats import trim1
def test_oneway(self):
    r1 = self.res_oneway
    r2s = self.res_2s
    res_bfm = self.res_bfm
    res_wa = self.res_wa
    res_fa = self.res_fa
    m = [x_i.mean() for x_i in self.x]
    assert_allclose(m, self.res_m, rtol=1e-13)
    resg = smo.anova_oneway(self.x, use_var='unequal', trim_frac=1 / 13)
    assert_allclose(resg.pvalue, r1.p_value, rtol=1e-13)
    assert_allclose(resg.df, [r1.df1, r1.df2], rtol=1e-13)
    resg = smo.anova_oneway(self.x[:2], use_var='unequal', trim_frac=1 / 13)
    assert_allclose(resg.pvalue, r2s.p_value, rtol=1e-13)
    assert_allclose(resg.df, [1, r2s.df], rtol=1e-13)
    res = smo.anova_oneway(self.x, use_var='bf')
    assert_allclose(res[0], res_bfm.statistic, rtol=1e-13)
    assert_allclose(res.pvalue2, res_bfm.p_value, rtol=1e-13)
    assert_allclose(res.df2, res_bfm.parameter, rtol=1e-13)
    res = smo.anova_oneway(self.x, use_var='unequal')
    assert_allclose(res.statistic, res_wa.statistic, rtol=1e-13)
    assert_allclose(res.pvalue, res_wa.p_value, rtol=1e-13)
    assert_allclose(res.df, res_wa.parameter, rtol=1e-13)
    res = smo.anova_oneway(self.x, use_var='equal')
    assert_allclose(res.statistic, res_fa.statistic, rtol=1e-13)
    assert_allclose(res.pvalue, res_fa.p_value, rtol=1e-13)
    assert_allclose(res.df, res_fa.parameter, rtol=1e-13)
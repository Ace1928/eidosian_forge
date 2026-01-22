import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
def test_postestimation(self):
    res1 = self.res1
    res2 = self.res2
    resid_prob = res1.resid_prob
    assert_allclose(resid_prob[:len(res2.resid_prob)], res2.resid_prob, atol=0.0001)
    stats_prob = [resid_prob.mean(), resid_prob.min(), resid_prob.max(), resid_prob.var(ddof=1)]
    assert_allclose(stats_prob, res2.resid_prob_stats, atol=1e-05)
    chi2 = 20.958760713111
    df = 17
    p_value = 0.2281403796588
    import statsmodels.stats.diagnostic_gen as dia
    fitted = res1.predict()
    y_dummy = (res1.model.endog[:, None] == np.arange(3)).astype(int)
    sv = (fitted * np.arange(1, 3 + 1)).sum(1)
    dt = dia.test_chisquare_binning(y_dummy, fitted, sort_var=sv, bins=10, df=None, ordered=True, sort_method='stable')
    assert_allclose(dt.statistic, chi2, rtol=5e-05)
    assert_allclose(dt.pvalue, p_value, rtol=0.0001)
    assert_equal(dt.df, df)
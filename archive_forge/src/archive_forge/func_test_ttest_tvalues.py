from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
def test_ttest_tvalues(self):
    smt.check_ttest_tvalues(self.results)
    res = self.results
    mat = np.eye(len(res.params))
    tt = res.t_test(mat[0])
    string_confint = lambda alpha: '[{:4.3F}      {:4.3F}]'.format(alpha / 2, 1 - alpha / 2)
    summ = tt.summary()
    assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)
    assert_(string_confint(0.05) in str(summ))
    summ = tt.summary(alpha=0.1)
    ss = '[0.05       0.95]'
    assert_(ss in str(summ))
    summf = tt.summary_frame(alpha=0.1)
    pvstring_use_t = 'P>|z|' if res.use_t is False else 'P>|t|'
    tstring_use_t = 'z' if res.use_t is False else 't'
    cols = ['coef', 'std err', tstring_use_t, pvstring_use_t, 'Conf. Int. Low', 'Conf. Int. Upp.']
    assert_array_equal(summf.columns.values, cols)
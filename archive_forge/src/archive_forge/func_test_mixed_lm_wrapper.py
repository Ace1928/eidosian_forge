from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def test_mixed_lm_wrapper():
    np.random.seed(2410)
    exog = np.random.normal(size=(300, 4))
    exog_re = np.random.normal(size=300)
    groups = np.kron(np.arange(100), [1, 1, 1])
    g_errors = exog_re * np.kron(np.random.normal(size=100), [1, 1, 1])
    endog = exog.sum(1) + g_errors + np.random.normal(size=300)
    df = pd.DataFrame({'endog': endog})
    for k in range(exog.shape[1]):
        df['exog%d' % k] = exog[:, k]
    df['exog_re'] = exog_re
    fml = 'endog ~ 0 + exog0 + exog1 + exog2 + exog3'
    re_fml = '~ exog_re'
    mod2 = MixedLM.from_formula(fml, df, re_formula=re_fml, groups=groups)
    result = mod2.fit()
    result.summary()
    xnames = ['exog0', 'exog1', 'exog2', 'exog3']
    re_names = ['Group', 'exog_re']
    re_names_full = ['Group Var', 'Group x exog_re Cov', 'exog_re Var']
    assert_(mod2.data.xnames == xnames)
    assert_(mod2.data.exog_re_names == re_names)
    assert_(mod2.data.exog_re_names_full == re_names_full)
    params = result.params
    assert_(params.index.tolist() == xnames + re_names_full)
    bse = result.bse
    assert_(bse.index.tolist() == xnames + re_names_full)
    tvalues = result.tvalues
    assert_(tvalues.index.tolist() == xnames + re_names_full)
    cov_params = result.cov_params()
    assert_(cov_params.index.tolist() == xnames + re_names_full)
    assert_(cov_params.columns.tolist() == xnames + re_names_full)
    fe = result.fe_params
    assert_(fe.index.tolist() == xnames)
    bse_fe = result.bse_fe
    assert_(bse_fe.index.tolist() == xnames)
    cov_re = result.cov_re
    assert_(cov_re.index.tolist() == re_names)
    assert_(cov_re.columns.tolist() == re_names)
    cov_re_u = result.cov_re_unscaled
    assert_(cov_re_u.index.tolist() == re_names)
    assert_(cov_re_u.columns.tolist() == re_names)
    bse_re = result.bse_re
    assert_(bse_re.index.tolist() == re_names_full)
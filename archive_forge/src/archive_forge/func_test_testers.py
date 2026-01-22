import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import statsmodels.api as sm
from statsmodels.stats import knockoff_regeffects as kr
from statsmodels.stats._knockoff import (RegressionFDR,
@pytest.mark.parametrize('p', [49, 50])
@pytest.mark.parametrize('tester', [kr.CorrelationEffects(), kr.ForwardEffects(pursuit=False), kr.ForwardEffects(pursuit=True), kr.OLSEffects(), kr.RegModelEffects(sm.OLS), kr.RegModelEffects(sm.OLS, True, fit_kws={'L1_wt': 0, 'alpha': 1})])
@pytest.mark.parametrize('method', ['equi', 'sdp'])
def test_testers(p, tester, method):
    if method == 'sdp' and (not has_cvxopt):
        return
    np.random.seed(2432)
    n = 200
    y = np.random.normal(size=n)
    x = np.random.normal(size=(n, p))
    kn = RegressionFDR(y, x, tester, design_method=method)
    assert_equal(len(kn.stats), p)
    assert_equal(len(kn.fdr), p)
    kn.summary()
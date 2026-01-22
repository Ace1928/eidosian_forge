import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import statsmodels.api as sm
from statsmodels.stats import knockoff_regeffects as kr
from statsmodels.stats._knockoff import (RegressionFDR,
@pytest.mark.slow
@pytest.mark.parametrize('method', ['equi', 'sdp'])
@pytest.mark.parametrize('tester,n,p,es', [[kr.CorrelationEffects(), 300, 100, 6], [kr.ForwardEffects(pursuit=False), 300, 100, 3.5], [kr.ForwardEffects(pursuit=True), 300, 100, 3.5], [kr.OLSEffects(), 3000, 200, 3.5]])
def test_sim(method, tester, n, p, es):
    if method == 'sdp' and (not has_cvxopt):
        return
    np.random.seed(43234)
    npos = 30
    target_fdr = 0.2
    nrep = 10
    if method == 'sdp' and (not has_cvxopt):
        return
    fdr, power = (0, 0)
    for k in range(nrep):
        x = np.random.normal(size=(n, p))
        x /= np.sqrt(np.sum(x * x, 0))
        coeff = es * (-1) ** np.arange(npos)
        y = np.dot(x[:, 0:npos], coeff) + np.random.normal(size=n)
        kn = RegressionFDR(y, x, tester)
        tr = kn.threshold(target_fdr)
        cp = np.sum(kn.stats >= tr)
        fp = np.sum(kn.stats[npos:] >= tr)
        fdr += fp / max(cp, 1)
        power += np.mean(kn.stats[0:npos] >= tr)
        estimated_fdr = np.sum(kn.stats <= -tr) / (1 + np.sum(kn.stats >= tr))
        assert_equal(estimated_fdr < target_fdr, True)
    power /= nrep
    fdr /= nrep
    assert_array_equal(power > 0.6, True)
    assert_array_equal(fdr < target_fdr + 0.1, True)
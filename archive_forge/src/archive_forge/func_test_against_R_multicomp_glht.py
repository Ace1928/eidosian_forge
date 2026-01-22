import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.stats._multicomp import _pvalue_dunnett, DunnettResult
@pytest.mark.parametrize('case', [case_1, case_2, case_3, case_4])
@pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
def test_against_R_multicomp_glht(self, case, alternative):
    rng = np.random.default_rng(189117774084579816190295271136455278291)
    samples = case['samples']
    control = case['control']
    alternatives = {'less': 'less', 'greater': 'greater', 'two-sided': 'twosided'}
    p_ref = case['pvalues'][alternative.replace('-', '')]
    res = stats.dunnett(*samples, control=control, alternative=alternative, random_state=rng)
    assert_allclose(res.pvalue, p_ref, rtol=0.005, atol=0.0001)
    ci_ref = case['cis'][alternatives[alternative]]
    if alternative == 'greater':
        ci_ref = [ci_ref, np.inf]
    elif alternative == 'less':
        ci_ref = [-np.inf, ci_ref]
    assert res._ci is None
    assert res._ci_cl is None
    ci = res.confidence_interval(confidence_level=0.95)
    assert_allclose(ci.low, ci_ref[0], rtol=0.005, atol=1e-05)
    assert_allclose(ci.high, ci_ref[1], rtol=0.005, atol=1e-05)
    assert res._ci is ci
    assert res._ci_cl == 0.95
    ci_ = res.confidence_interval(confidence_level=0.95)
    assert ci_ is ci
import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('method', methods)
def test_rate_poisson_consistency(method):
    count, nobs = (15, 400)
    ci = confint_poisson(count, nobs, method=method)
    pv1 = smr.test_poisson(count, nobs, value=ci[0], method=method).pvalue
    pv2 = smr.test_poisson(count, nobs, value=ci[1], method=method).pvalue
    rtol = 1e-10
    if method in ['midp-c']:
        rtol = 1e-06
    assert_allclose(pv1, 0.05, rtol=rtol)
    assert_allclose(pv2, 0.05, rtol=rtol)
    pv1 = smr.test_poisson(count, nobs, value=ci[0], method=method, alternative='larger').pvalue
    pv2 = smr.test_poisson(count, nobs, value=ci[1], method=method, alternative='smaller').pvalue
    assert_allclose(pv1, 0.025, rtol=rtol)
    assert_allclose(pv2, 0.025, rtol=rtol)
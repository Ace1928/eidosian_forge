import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('method', methods_diff_ratio)
def test_rate_poisson_diff_ratio_consistency(method):
    count1, n1, count2, n2 = (30, 400 / 10, 7, 300 / 10)
    t1 = smr.test_poisson_2indep(count1, n1, count2, n2, method=method, compare='ratio')
    t2 = smr.test_poisson_2indep(count1, n1, count2, n2, method=method, compare='diff')
    assert_allclose(t1.tuple, t2.tuple, rtol=1e-13)
    t1 = smr.test_poisson_2indep(count1, n1, count2, n2, method=method, compare='ratio', alternative='larger')
    t2 = smr.test_poisson_2indep(count1, n1, count2, n2, method=method, compare='diff', alternative='larger')
    assert_allclose(t1.tuple, t2.tuple, rtol=1e-13)
    t1 = smr.test_poisson_2indep(count1, n1, count2, n2, method=method, compare='ratio', alternative='smaller')
    t2 = smr.test_poisson_2indep(count1, n1, count2, n2, method=method, compare='diff', alternative='smaller')
    assert_allclose(t1.tuple, t2.tuple, rtol=1e-13)
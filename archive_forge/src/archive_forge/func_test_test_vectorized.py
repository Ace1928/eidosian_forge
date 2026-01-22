import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('compare, meth', [('ratio', meth) for meth in method_names_poisson_2indep['test']['ratio']] + [('diff', meth) for meth in method_names_poisson_2indep['test']['diff']])
def test_test_vectorized(self, meth, compare):
    if 'etest' in meth:
        pytest.skip('nonequivalence etest not vectorized')
    count1, n1, count2, n2 = (60, 514.775, 40, 543.087)
    count1v = np.array([count1, count2])
    n1v = np.array([n1, n2])
    nfact = 1.0
    count2v = np.array([count2, count1 * nfact], dtype=int)
    n2v = np.array([n2, n1 * nfact])
    count1, n1, count2, n2 = (count1v, n1v, count2v, n2v)
    if compare == 'ratio':
        f = 1.0
        low, upp = (1 / f, f)
    else:
        v = 0.0
        low, upp = (-v, v)
    tst2 = nonequivalence_poisson_2indep(count1, n1, count2, n2, low, upp, method=meth, compare=compare)
    assert tst2.statistic.shape == (2,)
    assert tst2.pvalue.shape == (2,)
    if not ('cond' in meth or 'etest' in meth):
        tst = smr.test_poisson_2indep(count1, n1, count2, n2, method=meth, compare=compare, value=None, alternative='two-sided')
        assert_allclose(tst2.pvalue, tst.pvalue, rtol=1e-12)
    if compare == 'ratio':
        f = 1.5
        low, upp = (1 / f, f)
    else:
        v = 0.5
        low, upp = (-v, v)
    tst0 = smr.tost_poisson_2indep(count1[0], n1[0], count2[0], n2[0], low, upp, method=meth, compare=compare)
    tst1 = smr.tost_poisson_2indep(count1[1], n1[1], count2[1], n2[1], low, upp, method=meth, compare=compare)
    tst2 = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp, method=meth, compare=compare)
    assert tst2.statistic.shape == (2,)
    assert tst2.pvalue.shape == (2,)
    assert_allclose(tst2.statistic[0], tst0.statistic, rtol=1e-12)
    assert_allclose(tst2.pvalue[0], tst0.pvalue, rtol=1e-12)
    assert_allclose(tst2.statistic[1], tst1.statistic, rtol=1e-12)
    assert_allclose(tst2.pvalue[1], tst1.pvalue, rtol=1e-12)
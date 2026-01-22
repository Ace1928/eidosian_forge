import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
@pytest.mark.xfail(reason='Known failure on modern SciPy >= 0.10', strict=True)
def test_power_solver_warn():
    pow_ = 0.6921941124382421
    nip = smp.NormalIndPower()
    nip.start_bqexp['nobs1'] = {'upp': 50, 'low': -20}
    val = nip.solve_power(0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative='larger')
    assert_almost_equal(val, 1600, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 3)
    nip.start_ttp['nobs1'] = np.nan
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    assert_warns(ConvergenceWarning, nip.solve_power, 0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative='larger')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val = nip.solve_power(0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative='larger')
        assert_equal(nip.cache_fit_res[0], 0)
        assert_equal(len(nip.cache_fit_res), 3)
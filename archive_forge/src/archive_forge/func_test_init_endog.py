import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def test_init_endog():
    index = pd.date_range(start='1950-01-01', periods=10, freq='D')
    endog = [np.ones(10), pd.Series(np.ones(10), index=index), np.ones((10, 1)), pd.DataFrame(np.ones((10, 1)), index=index)]
    for _endog in endog:
        mod = markov_switching.MarkovSwitching(_endog, k_regimes=2)
        assert_equal(mod.nobs, 10)
        assert_equal(mod.endog, _endog.squeeze())
        assert_equal(mod.k_regimes, 2)
        assert_equal(mod.tvtp, False)
        assert_equal(mod.k_tvtp, 0)
        assert_equal(mod.k_params, 2)
    endog = np.ones(10)
    assert_raises(ValueError, markov_switching.MarkovSwitching, endog, k_regimes=1)
    endog = np.ones((10, 2))
    assert_raises(ValueError, markov_switching.MarkovSwitching, endog, k_regimes=2)
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def test_init_exog_tvtp():
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=exog_tvtp)
    assert_equal(mod.tvtp, True)
    assert_equal(mod.k_tvtp, 2)
    exog_tvtp = np.c_[np.ones((11, 1)), (np.arange(11) + 1)[:, np.newaxis]]
    assert_raises(ValueError, markov_switching.MarkovSwitching, endog, k_regimes=2, exog_tvtp=exog_tvtp)
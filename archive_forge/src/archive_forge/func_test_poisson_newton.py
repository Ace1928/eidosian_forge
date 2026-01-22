from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_poisson_newton():
    nobs = 10000
    np.random.seed(987689)
    x = np.random.randn(nobs, 3)
    x = sm.add_constant(x, prepend=True)
    y_count = np.random.poisson(np.exp(x.sum(1)))
    mod = sm.Poisson(y_count, x)
    with pytest.warns(ConvergenceWarning):
        res = mod.fit(start_params=-np.ones(4), method='newton', disp=0)
    assert_(not res.mle_retvals['converged'])
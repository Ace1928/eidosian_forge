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
def test_predict_prob(self):
    res = self.res
    endog = res.model.endog
    freq = np.bincount(endog.astype(int))
    pr = res.predict(which='prob')
    pr2 = sm.distributions.genpoisson_p.pmf(np.arange(6)[:, None], res.predict(), res.params[-1], 1).T
    assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)
    expected = pr.sum(0)
    expected[-1] += pr.shape[0] - expected.sum()
    assert_allclose(freq.sum(), expected.sum(), rtol=1e-13)
    from scipy import stats
    chi2 = stats.chisquare(freq, expected)
    assert_allclose(chi2[:], (0.5511787456691261, 0.9901293016678583), rtol=0.01)
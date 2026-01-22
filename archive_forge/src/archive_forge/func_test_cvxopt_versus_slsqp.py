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
def test_cvxopt_versus_slsqp(self):
    self.alpha = 3.0 * np.array([0, 1, 1, 1.0])
    res_slsqp = Logit(self.data.endog, self.data.exog).fit_regularized(method='l1', alpha=self.alpha, disp=0, acc=1e-10, maxiter=1000, trim_mode='auto')
    res_cvxopt = Logit(self.data.endog, self.data.exog).fit_regularized(method='l1_cvxopt_cp', alpha=self.alpha, disp=0, abstol=1e-10, trim_mode='auto', auto_trim_tol=0.01, maxiter=1000)
    assert_almost_equal(res_slsqp.params, res_cvxopt.params, DECIMAL_4)
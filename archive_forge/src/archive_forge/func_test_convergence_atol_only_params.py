import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def test_convergence_atol_only_params(self):
    atol = 1e-08
    rtol = 0
    self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
    expected_iterations = self._when_converged(atol=atol, rtol=rtol, tol_criterion='params')
    actual_iterations = self.res.fit_history['iteration']
    assert_equal(expected_iterations, actual_iterations)
    assert_equal(len(self.res.fit_history['deviance']) - 2, actual_iterations)
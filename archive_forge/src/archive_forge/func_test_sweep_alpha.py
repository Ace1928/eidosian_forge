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
def test_sweep_alpha(self):
    for i in range(3):
        alpha = self.alphas[i, :]
        res2 = self.model.fit_regularized(method='l1', alpha=alpha, disp=0, acc=1e-10, trim_mode='off', maxiter=1000)
        assert_almost_equal(res2.params, self.res1.params[i], DECIMAL_4)
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
def test_non_binary():
    y = [1, 2, 1, 2, 1, 2]
    X = np.random.randn(6, 2)
    assert_raises(ValueError, Logit, y, X)
    y = [0, 1, 0, 0, 1, 0.5]
    assert_raises(ValueError, Probit, y, X)
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
def test_mnlogit_basinhopping():

    def callb(*args):
        return 1
    x = np.random.randint(0, 100, 1000)
    y = np.random.randint(0, 3, 1000)
    model = MNLogit(y, sm.add_constant(x))
    model.fit(method='basinhopping')
    model.fit(method='basinhopping', callback=callb)
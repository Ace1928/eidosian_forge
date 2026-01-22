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
def test_nodummy_dydxmean(self):
    me = self.res1.get_margeff(at='mean')
    assert_almost_equal(me.margeff, self.res2.margeff_nodummy_dydxmean, DECIMAL_4)
    assert_almost_equal(me.margeff_se, self.res2.margeff_nodummy_dydxmean_se, DECIMAL_4)
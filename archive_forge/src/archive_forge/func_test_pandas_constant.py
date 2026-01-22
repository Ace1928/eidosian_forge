from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
def test_pandas_constant(self):
    exog = self.data.exog.copy()
    exog['const'] = 1
    data = sm_data.handle_data(self.data.endog, exog)
    np.testing.assert_equal(data.k_constant, 1)
    np.testing.assert_equal(data.const_idx, 6)
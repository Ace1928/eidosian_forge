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
def test_aic_R(self):
    if self.res1.scale != 1:
        dof = 2
    else:
        dof = 0
    if isinstance(self.res1.model.family, sm.families.NegativeBinomial):
        llf = self.res1.model.family.loglike(self.res1.model.endog, self.res1.mu, self.res1.model.var_weights, self.res1.model.freq_weights, scale=1)
        aic = -2 * llf + 2 * (self.res1.df_model + 1)
    else:
        aic = self.res1.aic
    assert_almost_equal(aic + dof, self.res2.aic_R, self.decimal_aic_R)
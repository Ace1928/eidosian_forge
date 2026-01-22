import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_warnings_raised():
    weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
    weights = np.array(weights)
    gid = np.arange(1, 17 + 1) // 2
    cov_kwds = {'groups': gid, 'use_correction': False}
    with pytest.warns(SpecificationWarning):
        res1 = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), freq_weights=weights).fit(cov_type='cluster', cov_kwds=cov_kwds)
        res1.summary()
    with pytest.warns(SpecificationWarning):
        res1 = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), var_weights=weights).fit(cov_type='cluster', cov_kwds=cov_kwds)
        res1.summary()
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
def check_weights_as_formats(weights):
    res = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), freq_weights=weights).fit()
    assert isinstance(res._freq_weights, np.ndarray)
    assert isinstance(res._var_weights, np.ndarray)
    assert isinstance(res._iweights, np.ndarray)
    res = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), var_weights=weights).fit()
    assert isinstance(res._freq_weights, np.ndarray)
    assert isinstance(res._var_weights, np.ndarray)
    assert isinstance(res._iweights, np.ndarray)
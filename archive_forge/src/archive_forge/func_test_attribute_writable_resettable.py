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
def test_attribute_writable_resettable():
    data = sm.datasets.longley.load()
    endog, exog = (data.endog, data.exog)
    glm_model = sm.GLM(endog, exog)
    assert_equal(glm_model.family.link.power, 1.0)
    glm_model.family.link.power = 2.0
    assert_equal(glm_model.family.link.power, 2.0)
    glm_model2 = sm.GLM(endog, exog)
    assert_equal(glm_model2.family.link.power, 1.0)
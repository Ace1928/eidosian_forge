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
def test_compare_discrete(self):
    res1 = self.res1
    resd = self.resd
    assert_allclose(res1.llf, resd.llf, rtol=1e-10)
    score_obs1 = res1.model.score_obs(res1.params * 0.98)
    score_obsd = resd.model.score_obs(resd.params * 0.98)
    assert_allclose(score_obs1, score_obsd, rtol=1e-10)
    score1 = res1.model.score(res1.params * 0.98)
    assert_allclose(score1, score_obs1.sum(0), atol=1e-20)
    score0 = res1.model.score(res1.params)
    assert_allclose(score0, np.zeros(score_obs1.shape[1]), atol=5e-07)
    hessian1 = res1.model.hessian(res1.params * 0.98, observed=False)
    hessiand = resd.model.hessian(resd.params * 0.98)
    assert_allclose(hessian1, hessiand, rtol=1e-10)
    hessian1 = res1.model.hessian(res1.params * 0.98, observed=True)
    hessiand = resd.model.hessian(resd.params * 0.98)
    assert_allclose(hessian1, hessiand, rtol=1e-09)
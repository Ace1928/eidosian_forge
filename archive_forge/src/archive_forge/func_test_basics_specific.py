from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
def test_basics_specific(self):
    infl1 = self.infl1
    infl0 = self.infl0
    res1 = self.infl1.results
    res0 = self.infl0.results
    assert_allclose(res1.params, res1.params, rtol=1e-10)
    d1 = res1.model._deriv_mean_dparams(res1.params)
    d0 = res1.model._deriv_mean_dparams(res0.params)
    assert_allclose(d0, d1, rtol=1e-10)
    d1 = res1.model._deriv_score_obs_dendog(res1.params)
    d0 = res1.model._deriv_score_obs_dendog(res0.params)
    assert_allclose(d0, d1, rtol=1e-10)
    s1 = res1.model.score_obs(res1.params)
    s0 = res1.model.score_obs(res0.params)
    assert_allclose(s0, s1, rtol=1e-10)
    assert_allclose(infl0.hessian, infl1.hessian, rtol=1e-10)
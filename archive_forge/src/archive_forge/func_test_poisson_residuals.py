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
def test_poisson_residuals():
    nobs, k_exog = (100, 5)
    np.random.seed(987125)
    x = np.random.randn(nobs, k_exog - 1)
    x = add_constant(x)
    y_true = x.sum(1) / 2
    y = y_true + 2 * np.random.randn(nobs)
    exposure = 1 + np.arange(nobs) // 4
    yp = np.random.poisson(np.exp(y_true) * exposure)
    yp[10:15] += 10
    fam = sm.families.Poisson()
    mod_poi_e = GLM(yp, x, family=fam, exposure=exposure)
    res_poi_e = mod_poi_e.fit()
    mod_poi_w = GLM(yp / exposure, x, family=fam, var_weights=exposure)
    res_poi_w = mod_poi_w.fit()
    assert_allclose(res_poi_e.resid_response / exposure, res_poi_w.resid_response)
    assert_allclose(res_poi_e.resid_pearson, res_poi_w.resid_pearson)
    assert_allclose(res_poi_e.resid_deviance, res_poi_w.resid_deviance)
    assert_allclose(res_poi_e.resid_anscombe, res_poi_w.resid_anscombe)
    assert_allclose(res_poi_e.resid_anscombe_unscaled, res_poi_w.resid_anscombe)
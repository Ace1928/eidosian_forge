import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.meta_analysis import (
from .results import results_meta
def test_effectsize(self):
    res2 = self.results
    dta = (self.count1, self.nobs1, self.count2, self.nobs2)
    eff, var_eff = effectsize_2proportions(*dta)
    assert_allclose(eff, res2.y_rd, rtol=1e-13)
    assert_allclose(var_eff, res2.v_rd, rtol=1e-13)
    eff, var_eff = effectsize_2proportions(*dta, statistic='rr')
    assert_allclose(eff, res2.y_rr, rtol=1e-13)
    assert_allclose(var_eff, res2.v_rr, rtol=1e-13)
    eff, var_eff = effectsize_2proportions(*dta, statistic='or')
    assert_allclose(eff, res2.y_or, rtol=1e-13)
    assert_allclose(var_eff, res2.v_or, rtol=1e-13)
    eff, var_eff = effectsize_2proportions(*dta, statistic='as')
    assert_allclose(eff, res2.y_as, rtol=1e-13)
    assert_allclose(var_eff, res2.v_as, rtol=1e-13)
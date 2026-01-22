from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
@pytest.mark.smoke
@pytest.mark.matplotlib
def test_ordinal_plot(self, close_figures):
    family = families.Binomial()
    endog, exog, groups = load_data('gee_ordinal_1.csv', icept=False)
    va = cov_struct.GlobalOddsRatio('ordinal')
    mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
    rslt = mod.fit()
    fig = rslt.plot_distribution()
    assert_equal(isinstance(fig, plt.Figure), True)
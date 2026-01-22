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
def test_nominal_plot(self, close_figures):
    np.random.seed(34234)
    endog = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
    exog = np.ones((8, 2))
    exog[:, 1] = np.r_[1, 2, 1, 1, 2, 1, 2, 2]
    groups = np.arange(8)
    model = gee.NominalGEE(endog, exog, groups)
    result = model.fit(cov_type='naive', start_params=[3.295837, -2.197225])
    fig = result.plot_distribution()
    assert_equal(isinstance(fig, plt.Figure), True)
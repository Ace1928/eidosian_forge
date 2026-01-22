import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_added_variable_ols(self, close_figures):
    np.random.seed(3446)
    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    lin_pred = 4 + exog[:, 0] + 0.2 * exog[:, 1] ** 2
    endog = lin_pred + np.random.normal(size=n)
    model = sm.OLS(endog, exog)
    results = model.fit()
    fig = plot_added_variable(results, 0)
    ax = fig.get_axes()[0]
    ax.set_title('Added variable plot (OLS)')
    close_or_save(pdf, fig)
    close_figures()
import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_abline_model(self, close_figures):
    fig = abline_plot(model_results=self.mod)
    ax = fig.axes[0]
    ax.scatter(self.X[:, 1], self.y)
    close_or_save(pdf, fig)
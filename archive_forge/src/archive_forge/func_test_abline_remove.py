import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_abline_remove(self, close_figures):
    mod = self.mod
    intercept, slope = mod.params
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(self.X[:, 1], self.y)
    abline_plot(intercept=intercept, slope=slope, ax=ax)
    abline_plot(intercept=intercept, slope=2 * slope, ax=ax)
    lines = ax.get_lines()
    lines.pop(0).remove()
    close_or_save(pdf, fig)
import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
@pytest.mark.matplotlib
def test_with_ax(self, close_figures):
    plt = _import_mpl()
    fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert self.fig is fig
    assert self.ax is ax
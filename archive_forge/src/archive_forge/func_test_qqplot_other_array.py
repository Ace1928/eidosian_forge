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
def test_qqplot_other_array(self, close_figures):
    self.prbplt.qqplot(ax=self.ax, line=self.line, other=self.other_array, **self.plot_options)
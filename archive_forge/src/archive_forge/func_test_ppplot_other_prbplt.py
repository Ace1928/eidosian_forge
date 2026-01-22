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
def test_ppplot_other_prbplt(self, close_figures):
    self.prbplt.ppplot(ax=self.ax, line=self.line, other=self.other_prbplot, **self.plot_options)
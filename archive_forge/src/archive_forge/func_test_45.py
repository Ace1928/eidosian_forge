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
def test_45(self, close_figures):
    nchildren = len(self.ax.get_children())
    qqline(self.ax, '45')
    assert len(self.ax.get_children()) > nchildren
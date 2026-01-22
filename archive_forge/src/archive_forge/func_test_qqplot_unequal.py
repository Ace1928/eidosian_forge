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
def test_qqplot_unequal():
    rs = np.random.RandomState(0)
    data1 = rs.standard_normal(100)
    data2 = rs.standard_normal(200)
    fig1 = qqplot_2samples(data1, data2)
    fig2 = qqplot_2samples(data2, data1)
    x1, y1 = fig1.get_axes()[0].get_children()[0].get_data()
    x2, y2 = fig2.get_axes()[0].get_children()[0].get_data()
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)
    numobj1 = len(fig1.get_axes()[0].get_children())
    numobj2 = len(fig2.get_axes()[0].get_children())
    assert numobj1 == numobj2

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        qqplot(self.res, line='r')

    @pytest.mark.matplotlib
    def test_qqplot_2samples_prob_plot_obj(self, close_figures):
        for line in ['r', 'q', '45', 's']:
            qqplot_2samples(self.prbplt, self.other_prbplot, line=line)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_arrays(self, close_figures):
        for line in ['r', 'q', '45', 's']:
            qqplot_2samples(self.res, self.other_array, line=line)
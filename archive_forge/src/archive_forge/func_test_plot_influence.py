import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_plot_influence(self, close_figures):
    infl = self.res.get_influence()
    fig = influence_plot(self.res)
    assert_equal(isinstance(fig, plt.Figure), True)
    try:
        sizes = fig.axes[0].get_children()[0]._sizes
        ex = sm.add_constant(infl.cooks_distance[0])
        ssr = sm.OLS(sizes, ex).fit().ssr
        assert_array_less(ssr, 1e-12)
    except AttributeError:
        import warnings
        warnings.warn('test not compatible with matplotlib version')
    fig = influence_plot(self.res, criterion='DFFITS')
    assert_equal(isinstance(fig, plt.Figure), True)
    try:
        sizes = fig.axes[0].get_children()[0]._sizes
        ex = sm.add_constant(np.abs(infl.dffits[0]))
        ssr = sm.OLS(sizes, ex).fit().ssr
        assert_array_less(ssr, 1e-12)
    except AttributeError:
        pass
    assert_raises(ValueError, influence_plot, self.res, criterion='unknown')
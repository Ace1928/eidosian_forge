import numpy as np
from numpy.testing import assert_equal, assert_raises
from pandas import Series
import pytest
from statsmodels.graphics.factorplots import _recode, interaction_plot
@pytest.mark.matplotlib
@pytest.mark.parametrize('astype', ['str', 'int'])
def test_plot_pandas(self, astype, close_figures):
    weight = Series(self.weight, name='Weight').astype(astype)
    duration = Series(self.duration, name='Duration')
    days = Series(self.days, name='Days')
    fig = interaction_plot(weight, duration, days, markers=['D', '^'], ms=10)
    ax = fig.axes[0]
    trace = ax.get_legend().get_title().get_text()
    assert_equal(trace, 'Duration')
    assert_equal(ax.get_ylabel(), 'mean of Days')
    assert_equal(ax.get_xlabel(), 'Weight')
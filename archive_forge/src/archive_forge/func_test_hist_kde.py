import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.xfail(reason='Api changed in 3.6.0')
def test_hist_kde(self, ts):
    pytest.importorskip('scipy')
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.hist(logy=True, ax=ax)
    _check_ax_scales(ax, yaxis='log')
    xlabels = ax.get_xticklabels()
    _check_text_labels(xlabels, [''] * len(xlabels))
    ylabels = ax.get_yticklabels()
    _check_text_labels(ylabels, [''] * len(ylabels))
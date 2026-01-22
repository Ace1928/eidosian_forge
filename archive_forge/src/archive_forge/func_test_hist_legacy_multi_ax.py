import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_legacy_multi_ax(self, ts):
    fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2)
    _check_plot_works(ts.hist, figure=fig, ax=ax1, default_axes=True)
    _check_plot_works(ts.hist, figure=fig, ax=ax2, default_axes=True)
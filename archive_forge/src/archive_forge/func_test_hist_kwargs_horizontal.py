import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_kwargs_horizontal(self, ts):
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.hist(bins=5, ax=ax)
    ax = ts.plot.hist(orientation='horizontal', ax=ax)
    _check_text_labels(ax.xaxis.get_label(), 'Frequency')
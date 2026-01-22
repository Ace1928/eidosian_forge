import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_kwargs_align(self, ts):
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.hist(bins=5, ax=ax)
    ax = ts.plot.hist(align='left', stacked=True, ax=ax)
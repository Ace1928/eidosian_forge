import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_df_with_nonnumerics_no_bins(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'B', 'C', 'D'])
    df['E'] = ['x', 'y'] * 5
    _, ax = mpl.pyplot.subplots()
    ax = df.plot.hist(ax=ax)
    assert len(ax.patches) == 40
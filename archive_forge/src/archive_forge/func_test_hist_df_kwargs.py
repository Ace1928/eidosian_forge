import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_df_kwargs(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    _, ax = mpl.pyplot.subplots()
    ax = df.plot.hist(bins=5, ax=ax)
    assert len(ax.patches) == 10
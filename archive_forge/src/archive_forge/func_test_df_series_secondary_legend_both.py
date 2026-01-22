from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_df_series_secondary_legend_both(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 3)), columns=list('abc'))
    s = Series(np.random.default_rng(2).standard_normal(30), name='x')
    _, ax = mpl.pyplot.subplots()
    ax = df.plot(secondary_y=True, ax=ax)
    s.plot(legend=True, secondary_y=True, ax=ax)
    expected = ['a (right)', 'b (right)', 'c (right)', 'x (right)']
    _check_legend_labels(ax.left_ax, labels=expected)
    assert not ax.left_ax.get_yaxis().get_visible()
    assert ax.get_yaxis().get_visible()
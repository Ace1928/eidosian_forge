import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_df_legend_labels_secondary_y(self):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['a', 'b', 'c'])
    df2 = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['d', 'e', 'f'])
    df3 = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['g', 'h', 'i'])
    ax = df.plot(legend=True, secondary_y='b')
    _check_legend_labels(ax, labels=['a', 'b (right)', 'c'])
    ax = df2.plot(legend=False, ax=ax)
    _check_legend_labels(ax, labels=['a', 'b (right)', 'c'])
    ax = df3.plot(kind='bar', legend=True, secondary_y='h', ax=ax)
    _check_legend_labels(ax, labels=['a', 'b (right)', 'c', 'g', 'h (right)', 'i'])
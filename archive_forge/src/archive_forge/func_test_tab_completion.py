from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
def test_tab_completion(mframe):
    grp = mframe.groupby(level='second')
    results = {v for v in dir(grp) if not v.startswith('_')}
    expected = {'A', 'B', 'C', 'agg', 'aggregate', 'apply', 'boxplot', 'filter', 'first', 'get_group', 'groups', 'hist', 'indices', 'last', 'max', 'mean', 'median', 'min', 'ngroups', 'nth', 'ohlc', 'plot', 'prod', 'size', 'std', 'sum', 'transform', 'var', 'sem', 'count', 'nunique', 'head', 'describe', 'cummax', 'quantile', 'rank', 'cumprod', 'tail', 'resample', 'cummin', 'fillna', 'cumsum', 'cumcount', 'ngroup', 'all', 'shift', 'skew', 'take', 'pct_change', 'any', 'corr', 'corrwith', 'cov', 'dtypes', 'ndim', 'diff', 'idxmax', 'idxmin', 'ffill', 'bfill', 'rolling', 'expanding', 'pipe', 'sample', 'ewm', 'value_counts'}
    assert results == expected
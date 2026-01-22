import collections
from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
def test_value_counts_inferred(index_or_series, using_infer_string):
    klass = index_or_series
    s_values = ['a', 'b', 'b', 'b', 'b', 'c', 'd', 'd', 'a', 'a']
    s = klass(s_values)
    expected = Series([4, 3, 2, 1], index=['b', 'a', 'd', 'c'], name='count')
    tm.assert_series_equal(s.value_counts(), expected)
    if isinstance(s, Index):
        exp = Index(np.unique(np.array(s_values, dtype=np.object_)))
        tm.assert_index_equal(s.unique(), exp)
    else:
        exp = np.unique(np.array(s_values, dtype=np.object_))
        if using_infer_string:
            exp = array(exp)
        tm.assert_equal(s.unique(), exp)
    assert s.nunique() == 4
    hist = s.value_counts(sort=False).sort_values()
    expected = Series([3, 1, 4, 2], index=list('acbd'), name='count').sort_values()
    tm.assert_series_equal(hist, expected)
    hist = s.value_counts(ascending=True)
    expected = Series([1, 2, 3, 4], index=list('cdab'), name='count')
    tm.assert_series_equal(hist, expected)
    hist = s.value_counts(normalize=True)
    expected = Series([0.4, 0.3, 0.2, 0.1], index=['b', 'a', 'd', 'c'], name='proportion')
    tm.assert_series_equal(hist, expected)
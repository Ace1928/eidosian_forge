import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('method, expected_data', [['corr', [np.nan, 1.0, 1.0, 1]], ['cov', [np.nan, 0.5, 0.928571, 1.385714]]])
def test_pairwise_methods(self, method, expected_data):
    df = DataFrame({'A': ['a'] * 4, 'B': range(4)})
    result = getattr(df.groupby('A').ewm(com=1.0), method)()
    expected = DataFrame({'B': expected_data}, index=MultiIndex.from_tuples([('a', 0, 'B'), ('a', 1, 'B'), ('a', 2, 'B'), ('a', 3, 'B')], names=['A', None, None]))
    tm.assert_frame_equal(result, expected)
    expected = df.groupby('A')[['B']].apply(lambda x: getattr(x.ewm(com=1.0), method)())
    tm.assert_frame_equal(result, expected)
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize(('func', 'kwargs'), [('rolling', {'window': 2, 'min_periods': 1}), ('expanding', {})])
def test_groupby_rolling_sem(self, func, kwargs):
    df = DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2], ['b', 3]], columns=['a', 'b'])
    result = getattr(df.groupby('a'), func)(**kwargs).sem()
    expected = DataFrame({'a': [np.nan] * 5, 'b': [np.nan, 0.70711, np.nan, 0.70711, 0.70711]}, index=MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 2), ('b', 3), ('b', 4)], names=['a', None]))
    expected = expected.drop(columns='a')
    tm.assert_frame_equal(result, expected)
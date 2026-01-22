import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_concat_inner_sort(self, sort):
    df1 = DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [1, 2]}, columns=['b', 'a', 'c'])
    df2 = DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[3, 4])
    with tm.assert_produces_warning(None):
        result = pd.concat([df1, df2], sort=sort, join='inner', ignore_index=True)
    expected = DataFrame({'b': [1, 2, 3, 4], 'a': [1, 2, 1, 2]}, columns=['b', 'a'])
    if sort is True:
        expected = expected[['a', 'b']]
    tm.assert_frame_equal(result, expected)
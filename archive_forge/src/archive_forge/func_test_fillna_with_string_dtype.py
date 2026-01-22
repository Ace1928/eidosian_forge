import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method, expected', [('ffill', [None, 'a', 'a']), ('bfill', ['a', 'a', None])])
def test_fillna_with_string_dtype(method, expected):
    df = DataFrame({'a': pd.array([None, 'a', None], dtype='string'), 'b': [0, 0, 0]})
    grp = df.groupby('b')
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grp.fillna(method=method)
    expected = DataFrame({'a': pd.array(expected, dtype='string')})
    tm.assert_frame_equal(result, expected)
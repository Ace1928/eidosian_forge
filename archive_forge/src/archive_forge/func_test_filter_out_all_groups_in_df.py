from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_out_all_groups_in_df():
    df = DataFrame({'a': [1, 1, 2], 'b': [1, 2, 0]})
    res = df.groupby('a')
    res = res.filter(lambda x: x['b'].sum() > 5, dropna=False)
    expected = DataFrame({'a': [np.nan] * 3, 'b': [np.nan] * 3})
    tm.assert_frame_equal(expected, res)
    df = DataFrame({'a': [1, 1, 2], 'b': [1, 2, 0]})
    res = df.groupby('a')
    res = res.filter(lambda x: x['b'].sum() > 5, dropna=True)
    expected = DataFrame({'a': [], 'b': []}, dtype='int64')
    tm.assert_frame_equal(expected, res)
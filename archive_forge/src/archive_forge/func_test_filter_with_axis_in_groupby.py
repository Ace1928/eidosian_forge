from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_with_axis_in_groupby():
    index = pd.MultiIndex.from_product([range(10), [0, 1]])
    data = DataFrame(np.arange(100).reshape(-1, 20), columns=index, dtype='int64')
    msg = 'DataFrame.groupby with axis=1'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = data.groupby(level=0, axis=1)
    result = gb.filter(lambda x: x.iloc[0, 0] > 10)
    expected = data.iloc[:, 12:20]
    tm.assert_frame_equal(result, expected)
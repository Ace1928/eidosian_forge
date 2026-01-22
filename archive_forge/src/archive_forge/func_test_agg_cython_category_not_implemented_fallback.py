from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_agg_cython_category_not_implemented_fallback():
    df = DataFrame({'col_num': [1, 1, 2, 3]})
    df['col_cat'] = df['col_num'].astype('category')
    result = df.groupby('col_num').col_cat.first()
    expected = Series([1, 2, 3], index=Index([1, 2, 3], name='col_num'), name='col_cat', dtype=df['col_cat'].dtype)
    tm.assert_series_equal(result, expected)
    result = df.groupby('col_num').agg({'col_cat': 'first'})
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)
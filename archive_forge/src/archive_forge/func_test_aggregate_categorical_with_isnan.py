from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_aggregate_categorical_with_isnan():
    df = DataFrame({'A': [1, 1, 1, 1], 'B': [1, 2, 1, 2], 'numerical_col': [0.1, 0.2, np.nan, 0.3], 'object_col': ['foo', 'bar', 'foo', 'fee'], 'categorical_col': ['foo', 'bar', 'foo', 'fee']})
    df = df.astype({'categorical_col': 'category'})
    result = df.groupby(['A', 'B']).agg(lambda df: df.isna().sum())
    index = MultiIndex.from_arrays([[1, 1], [1, 2]], names=('A', 'B'))
    expected = DataFrame(data={'numerical_col': [1, 0], 'object_col': [0, 0], 'categorical_col': [0, 0]}, index=index)
    tm.assert_frame_equal(result, expected)
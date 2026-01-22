from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_categorical_dropna(observed, dropna):
    cat = Categorical([1, 2], categories=[1, 2, 3])
    df = DataFrame({'x': Categorical([1, 2], categories=[1, 2, 3]), 'y': [3, 4]})
    gb = df.groupby('x', observed=observed, dropna=dropna)
    result = gb.sum()
    if observed:
        expected = DataFrame({'y': [3, 4]}, index=cat)
    else:
        index = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected = DataFrame({'y': [3, 4, 0]}, index=index)
    expected.index.name = 'x'
    tm.assert_frame_equal(result, expected)
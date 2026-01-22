from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_categorical_index_and_columns(self, observed):
    columns = ['A', 'B', 'A', 'B']
    categories = ['B', 'A']
    data = np.array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]], int)
    cat_columns = CategoricalIndex(columns, categories=categories, ordered=True)
    df = DataFrame(data=data, columns=cat_columns)
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = df.groupby(axis=1, level=0, observed=observed).sum()
    expected_data = np.array([[4, 2], [4, 2], [4, 2], [4, 2], [4, 2]], int)
    expected_columns = CategoricalIndex(categories, categories=categories, ordered=True)
    expected = DataFrame(data=expected_data, columns=expected_columns)
    tm.assert_frame_equal(result, expected)
    df = DataFrame(data.T, index=cat_columns)
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(axis=0, level=0, observed=observed).sum()
    expected = DataFrame(data=expected_data.T, index=expected_columns)
    tm.assert_frame_equal(result, expected)
from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('ordered', [True, False])
def test_sort2(sort, ordered):
    df = DataFrame([['(7.5, 10]', 10, 10], ['(7.5, 10]', 8, 20], ['(2.5, 5]', 5, 30], ['(5, 7.5]', 6, 40], ['(2.5, 5]', 4, 50], ['(0, 2.5]', 1, 60], ['(5, 7.5]', 7, 70]], columns=['range', 'foo', 'bar'])
    df['range'] = Categorical(df['range'], ordered=ordered)
    result = df.groupby('range', sort=sort, observed=False).first()
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = ['(0, 2.5]', '(2.5, 5]', '(5, 7.5]', '(7.5, 10]']
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = ['(7.5, 10]', '(2.5, 5]', '(5, 7.5]', '(0, 2.5]']
    expected = DataFrame(data_values, columns=['foo', 'bar'], index=CategoricalIndex(index_values, name='range', ordered=ordered))
    tm.assert_frame_equal(result, expected)
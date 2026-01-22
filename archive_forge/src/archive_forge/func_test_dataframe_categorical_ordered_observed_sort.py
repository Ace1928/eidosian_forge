from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('ordered', [True, False])
@pytest.mark.parametrize('observed', [True, False])
@pytest.mark.parametrize('sort', [True, False])
def test_dataframe_categorical_ordered_observed_sort(ordered, observed, sort):
    label = Categorical(['d', 'a', 'b', 'a', 'd', 'b'], categories=['a', 'b', 'missing', 'd'], ordered=ordered)
    val = Series(['d', 'a', 'b', 'a', 'd', 'b'])
    df = DataFrame({'label': label, 'val': val})
    result = df.groupby('label', observed=observed, sort=sort)['val'].aggregate('first')
    label = Series(result.index.array, dtype='object')
    aggr = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = 'missing'
    if not all(label == aggr):
        msg = f'Labels and aggregation results not consistently sorted\nfor (ordered={ordered}, observed={observed}, sort={sort})\nResult:\n{result}'
        assert False, msg
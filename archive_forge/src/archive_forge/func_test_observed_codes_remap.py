from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_observed_codes_remap(observed):
    d = {'C1': [3, 3, 4, 5], 'C2': [1, 2, 3, 4], 'C3': [10, 100, 200, 34]}
    df = DataFrame(d)
    values = pd.cut(df['C1'], [1, 2, 3, 6])
    values.name = 'cat'
    groups_double_key = df.groupby([values, 'C2'], observed=observed)
    idx = MultiIndex.from_arrays([values, [1, 2, 3, 4]], names=['cat', 'C2'])
    expected = DataFrame({'C1': [3.0, 3.0, 4.0, 5.0], 'C3': [10.0, 100.0, 200.0, 34.0]}, index=idx)
    if not observed:
        expected = cartesian_product_for_groupers(expected, [values.values, [1, 2, 3, 4]], ['cat', 'C2'])
    result = groups_double_key.agg('mean')
    tm.assert_frame_equal(result, expected)
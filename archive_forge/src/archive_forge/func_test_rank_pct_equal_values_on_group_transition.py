from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('use_nan', [True, False])
def test_rank_pct_equal_values_on_group_transition(use_nan):
    fill_value = np.nan if use_nan else 3
    df = DataFrame([[-1, 1], [-1, 2], [1, fill_value], [-1, fill_value]], columns=['group', 'val'])
    result = df.groupby(['group'])['val'].rank(method='dense', pct=True)
    if use_nan:
        expected = Series([0.5, 1, np.nan, np.nan], name='val')
    else:
        expected = Series([1 / 3, 2 / 3, 1, 1], name='val')
    tm.assert_series_equal(result, expected)
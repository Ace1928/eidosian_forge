from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_column_select_via_attr(self, df):
    result = df.groupby('A').C.sum()
    expected = df.groupby('A')['C'].sum()
    tm.assert_series_equal(result, expected)
    df['mean'] = 1.5
    result = df.groupby('A').mean(numeric_only=True)
    expected = df.groupby('A')[['C', 'D', 'mean']].agg('mean')
    tm.assert_frame_equal(result, expected)
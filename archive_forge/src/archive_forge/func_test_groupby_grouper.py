from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_grouper(self, df):
    grouped = df.groupby('A')
    msg = 'DataFrameGroupBy.grouper is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouper = grouped.grouper
    result = df.groupby(grouper).mean(numeric_only=True)
    expected = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)
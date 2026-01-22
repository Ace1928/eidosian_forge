from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_iter_rolling_on_dataframe_unordered():
    df = DataFrame({'a': ['x', 'y', 'x'], 'b': [0, 1, 2]})
    results = list(df.groupby('a').rolling(2))
    expecteds = [df.iloc[idx, [1]] for idx in [[0], [0, 2], [1]]]
    for result, expected in zip(results, expecteds):
        tm.assert_frame_equal(result, expected)
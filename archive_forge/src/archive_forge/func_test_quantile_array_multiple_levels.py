import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_array_multiple_levels():
    df = DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5], 'c': ['a', 'a', 'a'], 'd': ['a', 'a', 'b']})
    result = df.groupby(['c', 'd']).quantile([0.25, 0.75])
    index = pd.MultiIndex.from_tuples([('a', 'a', 0.25), ('a', 'a', 0.75), ('a', 'b', 0.25), ('a', 'b', 0.75)], names=['c', 'd', None])
    expected = DataFrame({'A': [0.25, 0.75, 2.0, 2.0], 'B': [3.25, 3.75, 5.0, 5.0]}, index=index)
    tm.assert_frame_equal(result, expected)
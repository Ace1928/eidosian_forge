import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_with_duplicate_columns2(self):
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(5), 'B': np.random.default_rng(2).standard_normal(5), 'C': np.random.default_rng(2).standard_normal(5), 'D': ['a', 'b', 'c', 'd', 'e']})
    expected = df.take([0, 1, 1], axis=1)
    df2 = df.take([2, 0, 1, 2, 1], axis=1)
    result = df2.drop('C', axis=1)
    tm.assert_frame_equal(result, expected)
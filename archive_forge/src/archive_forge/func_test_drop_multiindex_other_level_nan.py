import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_multiindex_other_level_nan(self):
    df = DataFrame({'A': ['one', 'one', 'two', 'two'], 'B': [np.nan, 0.0, 1.0, 2.0], 'C': ['a', 'b', 'c', 'c'], 'D': [1, 2, 3, 4]}).set_index(['A', 'B', 'C']).sort_index()
    result = df.drop('c', level='C')
    expected = DataFrame([2, 1], columns=['D'], index=MultiIndex.from_tuples([('one', 0.0, 'b'), ('one', np.nan, 'a')], names=['A', 'B', 'C']))
    tm.assert_frame_equal(result, expected)
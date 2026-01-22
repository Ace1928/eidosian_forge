from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.parametrize('level', [0, 1])
def test_unstack_mixed_extension_types(self, level):
    index = MultiIndex.from_tuples([('A', 0), ('A', 1), ('B', 1)], names=['a', 'b'])
    df = DataFrame({'A': pd.array([0, 1, None], dtype='Int64'), 'B': pd.Categorical(['a', 'a', 'b'])}, index=index)
    result = df.unstack(level=level)
    expected = df.astype(object).unstack(level=level)
    if level == 0:
        expected['A', 'B'] = expected['A', 'B'].fillna(pd.NA)
    else:
        expected['A', 0] = expected['A', 0].fillna(pd.NA)
    expected_dtypes = Series([df.A.dtype] * 2 + [df.B.dtype] * 2, index=result.columns)
    tm.assert_series_equal(result.dtypes, expected_dtypes)
    tm.assert_frame_equal(result.astype(object), expected)
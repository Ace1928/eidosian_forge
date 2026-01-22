import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_aggregation_multi_level_column():
    lst = [[True, True, True, False], [True, False, np.nan, False], [True, True, np.nan, False], [True, True, np.nan, False]]
    df = DataFrame(data=lst, columns=MultiIndex.from_tuples([('A', 0), ('A', 1), ('B', 0), ('B', 1)]))
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=1, axis=1)
    result = gb.sum(numeric_only=False)
    expected = DataFrame({0: [2.0, True, True, True], 1: [1, 0, 1, 1]})
    tm.assert_frame_equal(result, expected)
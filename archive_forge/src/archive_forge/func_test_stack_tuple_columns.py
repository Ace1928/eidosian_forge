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
def test_stack_tuple_columns(future_stack):
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=[('a', 1), ('a', 2), ('b', 1)])
    result = df.stack(future_stack=future_stack)
    expected = Series([1, 2, 3, 4, 5, 6, 7, 8, 9], index=MultiIndex(levels=[[0, 1, 2], [('a', 1), ('a', 2), ('b', 1)]], codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]))
    tm.assert_series_equal(result, expected)
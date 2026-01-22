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
def test_agg_split_object_part_datetime():
    df = DataFrame({'A': pd.date_range('2000', periods=4), 'B': ['a', 'b', 'c', 'd'], 'C': [1, 2, 3, 4], 'D': ['b', 'c', 'd', 'e'], 'E': pd.date_range('2000', periods=4), 'F': [1, 2, 3, 4]}).astype(object)
    result = df.groupby([0, 0, 0, 0]).min()
    expected = DataFrame({'A': [pd.Timestamp('2000')], 'B': ['a'], 'C': [1], 'D': ['b'], 'E': [pd.Timestamp('2000')], 'F': [1]}, index=np.array([0]), dtype=object)
    tm.assert_frame_equal(result, expected)
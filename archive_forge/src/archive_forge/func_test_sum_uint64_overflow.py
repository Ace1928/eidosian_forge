import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_sum_uint64_overflow():
    df = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=object)
    df = df + 9223372036854775807
    index = Index([9223372036854775808, 9223372036854775810, 9223372036854775812], dtype=np.uint64)
    expected = DataFrame({1: [9223372036854775809, 9223372036854775811, 9223372036854775813]}, index=index, dtype=object)
    expected.index.name = 0
    result = df.groupby(0).sum(numeric_only=False)
    tm.assert_frame_equal(result, expected)
    result2 = df.groupby(0).sum(numeric_only=True)
    expected2 = expected[[]]
    tm.assert_frame_equal(result2, expected2)
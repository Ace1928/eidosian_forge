from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_crossed_dtypes_weird_corner(self):
    columns = ['A', 'B', 'C', 'D']
    df1 = DataFrame({'A': np.array([1, 2, 3, 4], dtype='f8'), 'B': np.array([1, 2, 3, 4], dtype='i8'), 'C': np.array([1, 2, 3, 4], dtype='f8'), 'D': np.array([1, 2, 3, 4], dtype='i8')}, columns=columns)
    df2 = DataFrame({'A': np.array([1, 2, 3, 4], dtype='i8'), 'B': np.array([1, 2, 3, 4], dtype='f8'), 'C': np.array([1, 2, 3, 4], dtype='i8'), 'D': np.array([1, 2, 3, 4], dtype='f8')}, columns=columns)
    appended = concat([df1, df2], ignore_index=True)
    expected = DataFrame(np.concatenate([df1.values, df2.values], axis=0), columns=columns)
    tm.assert_frame_equal(appended, expected)
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=['a'])
    df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=['b'])
    result = concat([df, df2], keys=['one', 'two'], names=['first', 'second'])
    assert result.index.names == ('first', 'second')
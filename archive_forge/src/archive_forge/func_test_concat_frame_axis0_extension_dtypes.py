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
def test_concat_frame_axis0_extension_dtypes():
    df1 = DataFrame({'a': pd.array([1, 2, 3], dtype='Int64')})
    df2 = DataFrame({'a': np.array([4, 5, 6])})
    result = concat([df1, df2], ignore_index=True)
    expected = DataFrame({'a': [1, 2, 3, 4, 5, 6]}, dtype='Int64')
    tm.assert_frame_equal(result, expected)
    result = concat([df2, df1], ignore_index=True)
    expected = DataFrame({'a': [4, 5, 6, 1, 2, 3]}, dtype='Int64')
    tm.assert_frame_equal(result, expected)
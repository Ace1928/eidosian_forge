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
def test_concat_mixed_objs_index(self):
    index = date_range('01-Jan-2013', periods=10, freq='h')
    arr = np.arange(10, dtype='int64')
    s1 = Series(arr, index=index)
    s2 = Series(arr, index=index)
    df = DataFrame(arr.reshape(-1, 1), index=index)
    expected = DataFrame(np.tile(arr, 3).reshape(-1, 1), index=index.tolist() * 3, columns=[0])
    result = concat([s1, df, s2])
    tm.assert_frame_equal(result, expected)
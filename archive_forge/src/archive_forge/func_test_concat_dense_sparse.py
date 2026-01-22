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
def test_concat_dense_sparse():
    dtype = pd.SparseDtype(np.float64, None)
    a = Series(pd.arrays.SparseArray([1, None]), dtype=dtype)
    b = Series([1], dtype=float)
    expected = Series(data=[1, None, 1], index=[0, 1, 0]).astype(dtype)
    result = concat([a, b], axis=0)
    tm.assert_series_equal(result, expected)
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
def test_concat_sparse():
    a = Series(SparseArray([0, 1, 2]))
    expected = DataFrame(data=[[0, 0], [1, 1], [2, 2]]).astype(pd.SparseDtype(np.int64, 0))
    result = concat([a, a], axis=1)
    tm.assert_frame_equal(result, expected)
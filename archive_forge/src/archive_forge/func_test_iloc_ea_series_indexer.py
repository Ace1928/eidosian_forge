from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_ea_series_indexer(self):
    df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    indexer = Series([0, 1], dtype='Int64')
    row_indexer = Series([1], dtype='Int64')
    result = df.iloc[row_indexer, indexer]
    expected = DataFrame([[5, 6]], index=[1])
    tm.assert_frame_equal(result, expected)
    result = df.iloc[row_indexer.values, indexer.values]
    tm.assert_frame_equal(result, expected)
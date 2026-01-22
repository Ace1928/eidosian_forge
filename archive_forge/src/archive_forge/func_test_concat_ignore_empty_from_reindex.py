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
@td.skip_array_manager_invalid_test
def test_concat_ignore_empty_from_reindex():
    df1 = DataFrame({'a': [1], 'b': [pd.Timestamp('2012-01-01')]})
    df2 = DataFrame({'a': [2]})
    aligned = df2.reindex(columns=df1.columns)
    msg = 'The behavior of DataFrame concatenation with empty or all-NA entries'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = concat([df1, aligned], ignore_index=True)
    expected = df1 = DataFrame({'a': [1, 2], 'b': [pd.Timestamp('2012-01-01'), pd.NaT]})
    tm.assert_frame_equal(result, expected)
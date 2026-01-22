from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_setitem_dictionary_value(self):
    df = DataFrame({'x': [1, 2], 'y': [2, 2]})
    rhs = {'x': 9, 'y': 99}
    df.iloc[1] = rhs
    expected = DataFrame({'x': [1, 9], 'y': [2, 99]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'x': [1, 2], 'y': [2.0, 2.0]})
    df.iloc[1] = rhs
    expected = DataFrame({'x': [1, 9], 'y': [2.0, 99.0]})
    tm.assert_frame_equal(df, expected)
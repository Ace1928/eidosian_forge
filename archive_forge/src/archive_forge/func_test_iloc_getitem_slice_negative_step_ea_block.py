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
def test_iloc_getitem_slice_negative_step_ea_block(self):
    df = DataFrame({'A': [1, 2, 3]}, dtype='Int64')
    res = df.iloc[:, ::-1]
    tm.assert_frame_equal(res, df)
    df['B'] = 'foo'
    res = df.iloc[:, ::-1]
    expected = DataFrame({'B': df['B'], 'A': df['A']})
    tm.assert_frame_equal(res, expected)
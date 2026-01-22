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
def test_iloc_setitem_frame_duplicate_columns_multiple_blocks(self, using_array_manager):
    df = DataFrame([[0, 1], [2, 3]], columns=['B', 'B'])
    df.iloc[:, 0] = df.iloc[:, 0].astype('f8')
    if not using_array_manager:
        assert len(df._mgr.blocks) == 1
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.iloc[:, 0] = df.iloc[:, 0] + 0.5
    if not using_array_manager:
        assert len(df._mgr.blocks) == 2
    expected = df.copy()
    df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
    tm.assert_frame_equal(df, expected)
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
@pytest.mark.parametrize(['dtypes', 'init_value', 'expected_value'], [('int64', '0', 0), ('float', '1.2', 1.2)])
def test_iloc_setitem_dtypes_duplicate_columns(self, dtypes, init_value, expected_value):
    df = DataFrame([[init_value, 'str', 'str2']], columns=['a', 'b', 'b'], dtype=object)
    df.iloc[:, 0] = df.iloc[:, 0].astype(dtypes)
    expected_df = DataFrame([[expected_value, 'str', 'str2']], columns=['a', 'b', 'b'], dtype=object)
    tm.assert_frame_equal(df, expected_df)
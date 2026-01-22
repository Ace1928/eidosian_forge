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
def test_iloc_setitem_list_duplicate_columns(self):
    df = DataFrame([[0, 'str', 'str2']], columns=['a', 'b', 'b'])
    df.iloc[:, 2] = ['str3']
    expected = DataFrame([[0, 'str', 'str3']], columns=['a', 'b', 'b'])
    tm.assert_frame_equal(df, expected)
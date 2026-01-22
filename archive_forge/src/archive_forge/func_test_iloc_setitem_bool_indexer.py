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
@pytest.mark.parametrize('klass', [list, np.array])
def test_iloc_setitem_bool_indexer(self, klass):
    df = DataFrame({'flag': ['x', 'y', 'z'], 'value': [1, 3, 4]})
    indexer = klass([True, False, False])
    df.iloc[indexer, 1] = df.iloc[indexer, 1] * 2
    expected = DataFrame({'flag': ['x', 'y', 'z'], 'value': [2, 3, 4]})
    tm.assert_frame_equal(df, expected)
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
def test_iloc_getitem_with_duplicates(self):
    df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=list('ABC'), index=list('aab'))
    result = df.iloc[0]
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.values, df.values[0])
    result = df.T.iloc[:, 0]
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.values, df.values[0])
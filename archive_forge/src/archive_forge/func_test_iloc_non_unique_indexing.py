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
def test_iloc_non_unique_indexing(self):
    df = DataFrame({'A': [0.1] * 3000, 'B': [1] * 3000})
    idx = np.arange(30) * 99
    expected = df.iloc[idx]
    df3 = concat([df, 2 * df, 3 * df])
    result = df3.iloc[idx]
    tm.assert_frame_equal(result, expected)
    df2 = DataFrame({'A': [0.1] * 1000, 'B': [1] * 1000})
    df2 = concat([df2, 2 * df2, 3 * df2])
    with pytest.raises(KeyError, match='not in index'):
        df2.loc[idx]
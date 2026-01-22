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
def test_iloc_setitem_series_duplicate_columns(self):
    df = DataFrame(np.arange(8, dtype=np.int64).reshape(2, 4), columns=['A', 'B', 'A', 'B'])
    df.iloc[:, 0] = df.iloc[:, 0].astype(np.float64)
    assert df.dtypes.iloc[2] == np.int64
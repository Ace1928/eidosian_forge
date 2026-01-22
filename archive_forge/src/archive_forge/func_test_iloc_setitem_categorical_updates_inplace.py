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
def test_iloc_setitem_categorical_updates_inplace(self):
    cat = Categorical(['A', 'B', 'C'])
    df = DataFrame({1: cat, 2: [1, 2, 3]}, copy=False)
    assert tm.shares_memory(df[1], cat)
    df.iloc[:, 0] = cat[::-1]
    assert tm.shares_memory(df[1], cat)
    expected = Categorical(['C', 'B', 'A'], categories=['A', 'B', 'C'])
    tm.assert_categorical_equal(cat, expected)
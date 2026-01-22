import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_frozenset_unique_in_column(self):
    df = DataFrame([[1, 2, 3, 4]], columns=[frozenset(['KEY']), 'B', 'C', 'C'])
    result = df[frozenset(['KEY'])]
    expected = Series([1], name=frozenset(['KEY']))
    tm.assert_series_equal(result, expected)
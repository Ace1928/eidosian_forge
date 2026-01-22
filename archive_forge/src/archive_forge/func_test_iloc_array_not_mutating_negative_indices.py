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
def test_iloc_array_not_mutating_negative_indices(self):
    array_with_neg_numbers = np.array([1, 2, -1])
    array_copy = array_with_neg_numbers.copy()
    df = DataFrame({'A': [100, 101, 102], 'B': [103, 104, 105], 'C': [106, 107, 108]}, index=[1, 2, 3])
    df.iloc[array_with_neg_numbers]
    tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
    df.iloc[:, array_with_neg_numbers]
    tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
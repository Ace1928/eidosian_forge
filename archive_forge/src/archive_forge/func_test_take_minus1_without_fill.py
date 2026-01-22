import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_take_minus1_without_fill(self, index):
    if len(index) == 0:
        pytest.skip("Test doesn't make sense for empty index")
    result = index.take([0, 0, -1])
    expected = index.take([0, 0, len(index) - 1])
    tm.assert_index_equal(result, expected)
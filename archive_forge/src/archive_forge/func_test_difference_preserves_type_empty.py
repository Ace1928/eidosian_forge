from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_preserves_type_empty(self, index, sort):
    if not index.is_unique:
        pytest.skip('Not relevant since index is not unique')
    result = index.difference(index, sort=sort)
    expected = index[:0]
    tm.assert_index_equal(result, expected, exact=True)
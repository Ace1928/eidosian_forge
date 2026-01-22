from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_symmetric_difference_mi(self, sort):
    index1 = MultiIndex.from_tuples(zip(['foo', 'bar', 'baz'], [1, 2, 3]))
    index2 = MultiIndex.from_tuples([('foo', 1), ('bar', 3)])
    result = index1.symmetric_difference(index2, sort=sort)
    expected = MultiIndex.from_tuples([('bar', 2), ('baz', 3), ('bar', 3)])
    if sort is None:
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)
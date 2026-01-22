import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_object_timestamp_categories(self):
    cat = Categorical([Timestamp('2014-01-01')])
    result = cat.astype(object)
    expected = np.array([Timestamp('2014-01-01 00:00:00')], dtype='object')
    tm.assert_numpy_array_equal(result, expected)
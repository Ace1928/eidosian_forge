import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_object_datetime_categories(self):
    cat = Categorical(to_datetime(['2021-03-27', NaT]))
    result = cat.astype(object)
    expected = np.array([Timestamp('2021-03-27 00:00:00'), NaT], dtype='object')
    tm.assert_numpy_array_equal(result, expected)
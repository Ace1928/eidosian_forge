import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [period_range('2000', periods=4), IntervalIndex.from_breaks([1, 2, 3, 4])])
def test_values_object_extension_dtypes(self, data):
    result = Series(data).values
    expected = np.array(data.astype(object))
    tm.assert_numpy_array_equal(result, expected)
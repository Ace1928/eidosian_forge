import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_no_prefix_string_cats_contains_get_dummies_NaN_column():
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0], 'NaN': [0, 0, 1]})
    expected = DataFrame({'': ['a', 'b', 'NaN']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)
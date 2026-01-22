import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_no_prefix_string_cats_basic_mixed_bool_values():
    dummies = DataFrame({'a': [1, 0, 0, 1], 'b': [False, True, False, False], 'c': [0, 0, 1, 0]})
    expected = DataFrame({'': ['a', 'b', 'c', 'a']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)
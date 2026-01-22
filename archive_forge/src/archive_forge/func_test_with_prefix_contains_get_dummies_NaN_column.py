import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_with_prefix_contains_get_dummies_NaN_column():
    dummies = DataFrame({'col1_a': [1, 0, 0], 'col1_b': [0, 1, 0], 'col1_NaN': [0, 0, 1], 'col2_a': [0, 1, 0], 'col2_b': [0, 0, 0], 'col2_c': [0, 0, 1], 'col2_NaN': [1, 0, 0]})
    expected = DataFrame({'col1': ['a', 'b', 'NaN'], 'col2': ['NaN', 'a', 'c']})
    result = from_dummies(dummies, sep='_')
    tm.assert_frame_equal(result, expected)
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_no_prefix_mixed_cats_basic():
    dummies = DataFrame({1.23: [1, 0, 0, 0, 0], 'c': [0, 1, 0, 0, 0], 2: [0, 0, 1, 0, 0], False: [0, 0, 0, 1, 0], None: [0, 0, 0, 0, 1]})
    expected = DataFrame({'': [1.23, 'c', 2, False, None]}, dtype='object')
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)
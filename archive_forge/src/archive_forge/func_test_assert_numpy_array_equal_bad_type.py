import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_assert_numpy_array_equal_bad_type():
    expected = 'Expected type'
    with pytest.raises(AssertionError, match=expected):
        tm.assert_numpy_array_equal(1, 2)
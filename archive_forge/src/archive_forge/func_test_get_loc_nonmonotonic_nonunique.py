import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_nonmonotonic_nonunique(self):
    cidx = CategoricalIndex(list('abcb'))
    result = cidx.get_loc('b')
    expected = np.array([False, True, False, True], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)
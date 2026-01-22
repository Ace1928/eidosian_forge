from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
def test_mixed_integer_with_codes(self):
    values = np.array(['b', 1, 0, 'a'], dtype=object)
    codes = [0, 1, 2, 3, 0, -1, 1]
    result, result_codes = safe_sort(values, codes)
    expected = np.array([0, 1, 'a', 'b'], dtype=object)
    expected_codes = np.array([3, 1, 0, 2, 3, -1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    tm.assert_numpy_array_equal(result_codes, expected_codes)
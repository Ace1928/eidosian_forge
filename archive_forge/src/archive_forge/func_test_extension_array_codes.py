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
@pytest.mark.parametrize('verify', [True, False])
def test_extension_array_codes(self, verify):
    a = array([1, 3, 2], dtype='Int64')
    result, codes = safe_sort(a, [0, 1, -1, 2], use_na_sentinel=True, verify=verify)
    expected_values = array([1, 2, 3], dtype='Int64')
    expected_codes = np.array([0, 2, -1, 1], dtype=np.intp)
    tm.assert_extension_array_equal(result, expected_values)
    tm.assert_numpy_array_equal(codes, expected_codes)
from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_comparisons_coverage(self):
    rng = timedelta_range('1 days', periods=10)
    result = rng < rng[3]
    expected = np.array([True, True, True] + [False] * 7)
    tm.assert_numpy_array_equal(result, expected)
    result = rng == list(rng)
    exp = rng == rng
    tm.assert_numpy_array_equal(result, exp)
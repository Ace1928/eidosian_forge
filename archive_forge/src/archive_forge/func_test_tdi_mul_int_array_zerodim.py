from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tdi_mul_int_array_zerodim(self, box_with_array):
    rng5 = np.arange(5, dtype='int64')
    idx = TimedeltaIndex(rng5)
    expected = TimedeltaIndex(rng5 * 5)
    idx = tm.box_expected(idx, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = idx * np.array(5, dtype='int64')
    tm.assert_equal(result, expected)
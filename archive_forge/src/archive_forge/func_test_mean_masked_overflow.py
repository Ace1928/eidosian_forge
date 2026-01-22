from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_mean_masked_overflow(self):
    val = 100000000000000000
    n_elements = 100
    na = np.array([val] * n_elements)
    ser = Series([val] * n_elements, dtype='Int64')
    result_numpy = np.mean(na)
    result_masked = ser.mean()
    assert result_masked - result_numpy == 0
    assert result_masked == 1e+17
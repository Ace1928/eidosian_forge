import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
def test_astype_boolean():
    a = pd.array([1, 0, -1, 2, None], dtype='Int64')
    result = a.astype('boolean')
    expected = pd.array([True, False, True, True, None], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
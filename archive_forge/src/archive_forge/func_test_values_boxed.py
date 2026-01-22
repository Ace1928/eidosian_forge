import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_values_boxed():
    tuples = [(1, pd.Timestamp('2000-01-01')), (2, pd.NaT), (3, pd.Timestamp('2000-01-03')), (1, pd.Timestamp('2000-01-04')), (2, pd.Timestamp('2000-01-02')), (3, pd.Timestamp('2000-01-03'))]
    result = MultiIndex.from_tuples(tuples)
    expected = construct_1d_object_array_from_listlike(tuples)
    tm.assert_numpy_array_equal(result.values, expected)
    tm.assert_numpy_array_equal(result.values[:4], result[:4].values)
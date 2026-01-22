from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_mismatched_dtype(self):
    dti = date_range('2016-01-01', periods=3)
    pi = dti.to_period('D')
    pi2 = dti.to_period('W')
    expected = np.array([-1, -1, -1], dtype=np.intp)
    result = pi.get_indexer(dti)
    tm.assert_numpy_array_equal(result, expected)
    result = dti.get_indexer(pi)
    tm.assert_numpy_array_equal(result, expected)
    result = pi.get_indexer(pi2)
    tm.assert_numpy_array_equal(result, expected)
    result = pi.get_indexer_non_unique(dti)[0]
    tm.assert_numpy_array_equal(result, expected)
    result = dti.get_indexer_non_unique(pi)[0]
    tm.assert_numpy_array_equal(result, expected)
    result = pi.get_indexer_non_unique(pi2)[0]
    tm.assert_numpy_array_equal(result, expected)
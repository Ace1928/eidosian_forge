from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_mismatched_dtype_different_length(self, non_comparable_idx):
    dti = date_range('2016-01-01', periods=3)
    pi = dti.to_period('D')
    other = non_comparable_idx
    res = pi[:-1].get_indexer(other)
    expected = -np.ones(other.shape, dtype=np.intp)
    tm.assert_numpy_array_equal(res, expected)
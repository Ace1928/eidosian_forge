import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ignore_downcast_neg_to_unsigned():
    data = ['-1', 2, 3]
    expected = np.array([-1, 2, 3], dtype=np.int64)
    res = to_numeric(data, downcast='unsigned')
    tm.assert_numpy_array_equal(res, expected)
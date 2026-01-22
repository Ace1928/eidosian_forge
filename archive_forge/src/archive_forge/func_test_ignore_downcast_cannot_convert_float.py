import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
@pytest.mark.parametrize('downcast', ['integer', 'signed', 'unsigned'])
@pytest.mark.parametrize('data,expected', [(['1.1', 2, 3], np.array([1.1, 2, 3], dtype=np.float64)), ([10000.0, 20000, 3000, 40000.36, 50000, 50000.0], np.array([10000.0, 20000, 3000, 40000.36, 50000, 50000.0], dtype=np.float64))])
def test_ignore_downcast_cannot_convert_float(data, expected, downcast):
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)
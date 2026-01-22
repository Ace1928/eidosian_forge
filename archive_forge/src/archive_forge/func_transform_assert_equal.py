import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=[(lambda x: Index(x, name='idx'), tm.assert_index_equal), (lambda x: Series(x, name='ser'), tm.assert_series_equal), (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal)])
def transform_assert_equal(request):
    return request.param
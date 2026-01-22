import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_rsub_object(self, fixed_now_ts):
    index = pd.Index([Decimal(1), Decimal(2)])
    expected = pd.Index([Decimal(1), Decimal(0)])
    result = Decimal(2) - index
    tm.assert_index_equal(result, expected)
    result = np.array([Decimal(2), Decimal(2)]) - index
    tm.assert_index_equal(result, expected)
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        'foo' - index
    with pytest.raises(TypeError, match=msg):
        np.array([True, fixed_now_ts]) - index
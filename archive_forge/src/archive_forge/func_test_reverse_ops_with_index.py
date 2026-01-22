from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op, expected', [(ops.rand_, Series([False, False])), (ops.ror_, Series([True, True])), (ops.rxor, Series([True, True]))])
def test_reverse_ops_with_index(self, op, expected):
    ser = Series([True, False])
    idx = Index([False, True])
    result = op(ser, idx)
    tm.assert_series_equal(result, expected)
from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('data, expected_data', [([0, 1, 2], [0, 2, 4])])
def test_integer_array_add_list_like(box_pandas_1d_array, box_1d_array, data, expected_data):
    arr = array(data, dtype='Int64')
    container = box_pandas_1d_array(arr)
    left = container + box_1d_array(data)
    right = box_1d_array(data) + container
    if Series in [box_1d_array, box_pandas_1d_array]:
        cls = Series
    elif Index in [box_1d_array, box_pandas_1d_array]:
        cls = Index
    else:
        cls = array
    expected = cls(expected_data, dtype='Int64')
    tm.assert_equal(left, expected)
    tm.assert_equal(right, expected)
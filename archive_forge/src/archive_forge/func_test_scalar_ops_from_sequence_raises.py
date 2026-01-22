from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
@pytest.mark.parametrize('class_', [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion])
def test_scalar_ops_from_sequence_raises(class_):
    arr = class_([decimal.Decimal('1.0'), decimal.Decimal('2.0')])
    result = arr + arr
    expected = np.array([decimal.Decimal('2.0'), decimal.Decimal('4.0')], dtype='object')
    tm.assert_numpy_array_equal(result, expected)
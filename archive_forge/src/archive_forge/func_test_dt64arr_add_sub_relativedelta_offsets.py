from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array, unit):
    vec = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'), Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')]).as_unit(unit)
    vec = tm.box_expected(vec, box_with_array)
    vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec
    relative_kwargs = [('years', 2), ('months', 5), ('days', 3), ('hours', 5), ('minutes', 10), ('seconds', 2), ('microseconds', 5)]
    for i, (offset_unit, value) in enumerate(relative_kwargs):
        off = DateOffset(**{offset_unit: value})
        exp_unit = unit
        if offset_unit == 'microseconds' and unit != 'ns':
            exp_unit = 'us'
        expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec + off)
        expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec - off)
        off = DateOffset(**dict(relative_kwargs[:i + 1]))
        expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec + off)
        expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec - off)
        msg = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            off - vec
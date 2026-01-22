from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('op_name', list(_ops.keys()))
@pytest.mark.parametrize('value,val_type', [(2, 'scalar'), (1.5, 'floating'), (np.nan, 'floating'), ('foo', 'str'), (timedelta(3600), 'timedelta'), (Timedelta('5s'), 'timedelta'), (datetime(2014, 1, 1), 'timestamp'), (Timestamp('2014-01-01'), 'timestamp'), (Timestamp('2014-01-01', tz='UTC'), 'timestamp'), (Timestamp('2014-01-01', tz='US/Eastern'), 'timestamp'), (pytz.timezone('Asia/Tokyo').localize(datetime(2014, 1, 1)), 'timestamp')])
def test_nat_arithmetic_scalar(op_name, value, val_type):
    invalid_ops = {'scalar': {'right_div_left'}, 'floating': {'right_div_left', 'left_minus_right', 'right_minus_left', 'left_plus_right', 'right_plus_left'}, 'str': set(_ops.keys()), 'timedelta': {'left_times_right', 'right_times_left'}, 'timestamp': {'left_times_right', 'right_times_left', 'left_div_right', 'right_div_left'}}
    op = _ops[op_name]
    if op_name in invalid_ops.get(val_type, set()):
        if val_type == 'timedelta' and 'times' in op_name and isinstance(value, Timedelta):
            typs = '(Timedelta|NaTType)'
            msg = f"unsupported operand type\\(s\\) for \\*: '{typs}' and '{typs}'"
        elif val_type == 'str':
            msg = '|'.join(['can only concatenate str', 'unsupported operand type', "can't multiply sequence", "Can't convert 'NaTType'", 'must be str, not NaTType'])
        else:
            msg = 'unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            op(NaT, value)
    else:
        if val_type == 'timedelta' and 'div' in op_name:
            expected = np.nan
        else:
            expected = NaT
        assert op(NaT, value) is expected
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('op', series_transform_kernels)
def test_transform_partial_failure(op, request):
    if op in ('ffill', 'bfill', 'pad', 'backfill', 'shift'):
        request.applymarker(pytest.mark.xfail(reason=f'{op} is successful on any dtype'))
    ser = Series(3 * [object])
    if op in ('fillna', 'ngroup'):
        error = ValueError
        msg = 'Transform function failed'
    else:
        error = TypeError
        msg = '|'.join(["not supported between instances of 'type' and 'type'", 'unsupported operand type'])
    with pytest.raises(error, match=msg):
        ser.transform([op, 'shift'])
    with pytest.raises(error, match=msg):
        ser.transform({'A': op, 'B': 'shift'})
    with pytest.raises(error, match=msg):
        ser.transform({'A': [op], 'B': ['shift']})
    with pytest.raises(error, match=msg):
        ser.transform({'A': [op, 'shift'], 'B': [op]})
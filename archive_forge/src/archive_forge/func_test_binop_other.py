from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
@pytest.mark.parametrize('value, dtype', [(1, 'i8'), (1.0, 'f8'), (2 ** 63, 'f8'), (1j, 'complex128'), (2 ** 63, 'complex128'), (True, 'bool'), (np.timedelta64(20, 'ns'), '<m8[ns]'), (np.datetime64(20, 'ns'), '<M8[ns]')])
@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv, operator.mod, operator.pow], ids=lambda x: x.__name__)
def test_binop_other(self, op, value, dtype, switch_numexpr_min_elements):
    skip = {(operator.truediv, 'bool'), (operator.pow, 'bool'), (operator.add, 'bool'), (operator.mul, 'bool')}
    elem = DummyElement(value, dtype)
    df = DataFrame({'A': [elem.value, elem.value]}, dtype=elem.dtype)
    invalid = {(operator.pow, '<M8[ns]'), (operator.mod, '<M8[ns]'), (operator.truediv, '<M8[ns]'), (operator.mul, '<M8[ns]'), (operator.add, '<M8[ns]'), (operator.pow, '<m8[ns]'), (operator.mul, '<m8[ns]'), (operator.sub, 'bool'), (operator.mod, 'complex128')}
    if (op, dtype) in invalid:
        warn = None
        if dtype == '<M8[ns]' and op == operator.add or (dtype == '<m8[ns]' and op == operator.mul):
            msg = None
        elif dtype == 'complex128':
            msg = "ufunc 'remainder' not supported for the input types"
        elif op is operator.sub:
            msg = 'numpy boolean subtract, the `-` operator, is '
            if dtype == 'bool' and expr.USE_NUMEXPR and (switch_numexpr_min_elements == 0):
                warn = UserWarning
        else:
            msg = f'cannot perform __{op.__name__}__ with this index type: (DatetimeArray|TimedeltaArray)'
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(warn):
                op(df, elem.value)
    elif (op, dtype) in skip:
        if op in [operator.add, operator.mul]:
            if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                warn = UserWarning
            else:
                warn = None
            with tm.assert_produces_warning(warn):
                op(df, elem.value)
        else:
            msg = "operator '.*' not implemented for .* dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op(df, elem.value)
    else:
        with tm.assert_produces_warning(None):
            result = op(df, elem.value).dtypes
            expected = op(df, value).dtypes
        tm.assert_series_equal(result, expected)
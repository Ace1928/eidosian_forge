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
@pytest.mark.parametrize('op', ['add', 'and', pytest.param('div', marks=pytest.mark.xfail(raises=AttributeError, reason='__idiv__ not implemented')), 'floordiv', 'mod', 'mul', 'or', 'pow', 'sub', 'truediv', 'xor'])
def test_inplace_ops_identity2(self, op):
    df = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1, 2, 3]})
    operand = 2
    if op in ('and', 'or', 'xor'):
        df['a'] = [True, False, True]
    df_copy = df.copy()
    iop = f'__i{op}__'
    op = f'__{op}__'
    getattr(df, iop)(operand)
    expected = getattr(df_copy, op)(operand)
    tm.assert_frame_equal(df, expected)
    expected = id(df)
    assert id(df) == expected
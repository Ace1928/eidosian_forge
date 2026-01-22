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
@pytest.mark.parametrize('left, right', [('gt', 'lt'), ('lt', 'gt'), ('ge', 'le'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
def test_timestamp_compare(self, left, right):
    df = DataFrame({'dates1': pd.date_range('20010101', periods=10), 'dates2': pd.date_range('20010102', periods=10), 'intcol': np.random.default_rng(2).integers(1000000000, size=10), 'floatcol': np.random.default_rng(2).standard_normal(10), 'stringcol': [chr(100 + i) for i in range(10)]})
    df.loc[np.random.default_rng(2).random(len(df)) > 0.5, 'dates2'] = pd.NaT
    left_f = getattr(operator, left)
    right_f = getattr(operator, right)
    if left in ['eq', 'ne']:
        expected = left_f(df, pd.Timestamp('20010109'))
        result = right_f(pd.Timestamp('20010109'), df)
        tm.assert_frame_equal(result, expected)
    else:
        msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            left_f(df, pd.Timestamp('20010109'))
        with pytest.raises(TypeError, match=msg):
            right_f(pd.Timestamp('20010109'), df)
    if left in ['eq', 'ne']:
        expected = left_f(df, pd.Timestamp('nat'))
        result = right_f(pd.Timestamp('nat'), df)
        tm.assert_frame_equal(result, expected)
    else:
        msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'NaTType'"
        with pytest.raises(TypeError, match=msg):
            left_f(df, pd.Timestamp('nat'))
        with pytest.raises(TypeError, match=msg):
            right_f(pd.Timestamp('nat'), df)
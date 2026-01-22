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
@pytest.mark.parametrize('arg, arg2', [[{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}], [{'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}], [{'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}], [{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}, {'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}]])
def test_comparison_invalid(self, arg, arg2):
    x = DataFrame(arg)
    y = DataFrame(arg2)
    result = x == y
    expected = DataFrame({col: x[col] == y[col] for col in x.columns}, index=x.index, columns=x.columns)
    tm.assert_frame_equal(result, expected)
    result = x != y
    expected = DataFrame({col: x[col] != y[col] for col in x.columns}, index=x.index, columns=x.columns)
    tm.assert_frame_equal(result, expected)
    msgs = ['Invalid comparison between dtype=datetime64\\[ns\\] and ndarray', 'invalid type promotion', "The DTypes <class 'numpy.dtype\\[.*\\]'> and <class 'numpy.dtype\\[.*\\]'> do not have a common DType."]
    msg = '|'.join(msgs)
    with pytest.raises(TypeError, match=msg):
        x >= y
    with pytest.raises(TypeError, match=msg):
        x > y
    with pytest.raises(TypeError, match=msg):
        x < y
    with pytest.raises(TypeError, match=msg):
        x <= y
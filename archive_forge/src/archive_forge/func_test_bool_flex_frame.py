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
@pytest.mark.parametrize('op', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
def test_bool_flex_frame(self, op):
    data = np.random.default_rng(2).standard_normal((5, 3))
    other_data = np.random.default_rng(2).standard_normal((5, 3))
    df = DataFrame(data)
    other = DataFrame(other_data)
    ndim_5 = np.ones(df.shape + (1, 3))
    assert df.eq(df).values.all()
    assert not df.ne(df).values.any()
    f = getattr(df, op)
    o = getattr(operator, op)
    tm.assert_frame_equal(f(other), o(df, other))
    part_o = other.loc[3:, 1:].copy()
    rs = f(part_o)
    xp = o(df, part_o.reindex(index=df.index, columns=df.columns))
    tm.assert_frame_equal(rs, xp)
    tm.assert_frame_equal(f(other.values), o(df, other.values))
    tm.assert_frame_equal(f(0), o(df, 0))
    msg = 'Unable to coerce to Series/DataFrame'
    tm.assert_frame_equal(f(np.nan), o(df, np.nan))
    with pytest.raises(ValueError, match=msg):
        f(ndim_5)
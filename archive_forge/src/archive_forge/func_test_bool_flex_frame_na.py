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
def test_bool_flex_frame_na(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    df.loc[0, 0] = np.nan
    rs = df.eq(df)
    assert not rs.loc[0, 0]
    rs = df.ne(df)
    assert rs.loc[0, 0]
    rs = df.gt(df)
    assert not rs.loc[0, 0]
    rs = df.lt(df)
    assert not rs.loc[0, 0]
    rs = df.ge(df)
    assert not rs.loc[0, 0]
    rs = df.le(df)
    assert not rs.loc[0, 0]
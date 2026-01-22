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
def test_align_frame(self):
    rng = pd.period_range('1/1/2000', '1/1/2010', freq='Y')
    ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng)
    result = ts + ts[::2]
    expected = ts + ts
    expected.iloc[1::2] = np.nan
    tm.assert_frame_equal(result, expected)
    half = ts[::2]
    result = ts + half.take(np.random.default_rng(2).permutation(len(half)))
    tm.assert_frame_equal(result, expected)
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
def test_inplace_ops_alignment(self):
    columns = list('abcdefg')
    X_orig = DataFrame(np.arange(10 * len(columns)).reshape(-1, len(columns)), columns=columns, index=range(10))
    Z = 100 * X_orig.iloc[:, 1:-1].copy()
    block1 = list('bedcf')
    subs = list('bcdef')
    X = X_orig.copy()
    result1 = (X[block1] + Z).reindex(columns=subs)
    X[block1] += Z
    result2 = X.reindex(columns=subs)
    X = X_orig.copy()
    result3 = (X[block1] + Z[block1]).reindex(columns=subs)
    X[block1] += Z[block1]
    result4 = X.reindex(columns=subs)
    tm.assert_frame_equal(result1, result2)
    tm.assert_frame_equal(result1, result3)
    tm.assert_frame_equal(result1, result4)
    X = X_orig.copy()
    result1 = (X[block1] - Z).reindex(columns=subs)
    X[block1] -= Z
    result2 = X.reindex(columns=subs)
    X = X_orig.copy()
    result3 = (X[block1] - Z[block1]).reindex(columns=subs)
    X[block1] -= Z[block1]
    result4 = X.reindex(columns=subs)
    tm.assert_frame_equal(result1, result2)
    tm.assert_frame_equal(result1, result3)
    tm.assert_frame_equal(result1, result4)
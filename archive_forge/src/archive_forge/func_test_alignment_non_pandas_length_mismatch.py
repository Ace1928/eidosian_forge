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
@pytest.mark.parametrize('val', [[1, 2], (1, 2), np.array([1, 2]), range(1, 3)])
def test_alignment_non_pandas_length_mismatch(self, val):
    index = ['A', 'B', 'C']
    columns = ['X', 'Y', 'Z']
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
    align = DataFrame._align_for_op
    msg = 'Unable to coerce to Series, length must be 3: given 2'
    with pytest.raises(ValueError, match=msg):
        align(df, val, axis=0)
    with pytest.raises(ValueError, match=msg):
        align(df, val, axis=1)
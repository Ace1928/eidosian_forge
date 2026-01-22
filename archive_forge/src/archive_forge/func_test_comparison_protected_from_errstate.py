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
def test_comparison_protected_from_errstate(self):
    missing_df = DataFrame(np.ones((10, 4), dtype=np.float64), columns=Index(list('ABCD'), dtype=object))
    missing_df.loc[missing_df.index[0], 'A'] = np.nan
    with np.errstate(invalid='ignore'):
        expected = missing_df.values < 0
    with np.errstate(invalid='raise'):
        result = (missing_df < 0).values
    tm.assert_numpy_array_equal(result, expected)
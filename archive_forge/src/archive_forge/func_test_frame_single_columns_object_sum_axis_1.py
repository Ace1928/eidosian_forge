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
def test_frame_single_columns_object_sum_axis_1():
    data = {'One': Series(['A', 1.2, np.nan])}
    df = DataFrame(data)
    result = df.sum(axis=1)
    expected = Series(['A', 1.2, 0])
    tm.assert_series_equal(result, expected)
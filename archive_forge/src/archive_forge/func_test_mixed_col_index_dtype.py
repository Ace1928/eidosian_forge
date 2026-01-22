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
def test_mixed_col_index_dtype():
    df1 = DataFrame(columns=list('abc'), data=1.0, index=[0])
    df2 = DataFrame(columns=list('abc'), data=0.0, index=[0])
    df1.columns = df2.columns.astype('string')
    result = df1 + df2
    expected = DataFrame(columns=list('abc'), data=1.0, index=[0])
    tm.assert_frame_equal(result, expected)
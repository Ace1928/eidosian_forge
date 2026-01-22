import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_get_by_index():
    df = DataFrame({'A': ['S', 'W', 'W'], 'B': [1.0, 1.0, 2.0]})
    res = df.groupby('A').agg({'B': lambda x: x.get(x.index[-1])})
    expected = DataFrame({'A': ['S', 'W'], 'B': [1.0, 2.0]}).set_index('A')
    tm.assert_frame_equal(res, expected)
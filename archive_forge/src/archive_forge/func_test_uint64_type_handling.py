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
@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
@pytest.mark.parametrize('how', ['first', 'last', 'min', 'max', 'mean', 'median'])
def test_uint64_type_handling(dtype, how):
    df = DataFrame({'x': 6903052872240755750, 'y': [1, 2]})
    expected = df.groupby('y').agg({'x': how})
    df.x = df.x.astype(dtype)
    result = df.groupby('y').agg({'x': how})
    if how not in ('mean', 'median'):
        result.x = result.x.astype(np.int64)
    tm.assert_frame_equal(result, expected, check_exact=True)
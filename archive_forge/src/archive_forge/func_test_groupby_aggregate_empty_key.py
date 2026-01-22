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
@pytest.mark.parametrize('kwargs', [{'c': ['min']}, {'b': [], 'c': ['min']}])
def test_groupby_aggregate_empty_key(kwargs):
    df = DataFrame({'a': [1, 1, 2], 'b': [1, 2, 3], 'c': [1, 2, 4]})
    result = df.groupby('a').agg(kwargs)
    expected = DataFrame([1, 4], index=Index([1, 2], dtype='int64', name='a'), columns=MultiIndex.from_tuples([['c', 'min']]))
    tm.assert_frame_equal(result, expected)
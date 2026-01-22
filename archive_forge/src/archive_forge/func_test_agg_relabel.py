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
def test_agg_relabel(self):
    df = DataFrame({'group': ['a', 'a', 'b', 'b'], 'A': [0, 1, 2, 3], 'B': [5, 6, 7, 8]})
    result = df.groupby('group').agg(a_max=('A', 'max'), b_max=('B', 'max'))
    expected = DataFrame({'a_max': [1, 3], 'b_max': [6, 8]}, index=Index(['a', 'b'], name='group'), columns=['a_max', 'b_max'])
    tm.assert_frame_equal(result, expected)
    p98 = functools.partial(np.percentile, q=98)
    result = df.groupby('group').agg(b_min=('B', 'min'), a_min=('A', 'min'), a_mean=('A', 'mean'), a_max=('A', 'max'), b_max=('B', 'max'), a_98=('A', p98))
    expected = DataFrame({'b_min': [5, 7], 'a_min': [0, 2], 'a_mean': [0.5, 2.5], 'a_max': [1, 3], 'b_max': [6, 8], 'a_98': [0.98, 2.98]}, index=Index(['a', 'b'], name='group'), columns=['b_min', 'a_min', 'a_mean', 'a_max', 'b_max', 'a_98'])
    tm.assert_frame_equal(result, expected)
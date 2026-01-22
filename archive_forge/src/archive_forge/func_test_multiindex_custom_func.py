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
@pytest.mark.parametrize('func', [lambda s: s.mean(), lambda s: np.mean(s), lambda s: np.nanmean(s)])
def test_multiindex_custom_func(func):
    data = [[1, 4, 2], [5, 7, 1]]
    df = DataFrame(data, columns=MultiIndex.from_arrays([[1, 1, 2], [3, 4, 3]], names=['Sisko', 'Janeway']))
    result = df.groupby(np.array([0, 1])).agg(func)
    expected_dict = {(1, 3): {0: 1.0, 1: 5.0}, (1, 4): {0: 4.0, 1: 7.0}, (2, 3): {0: 2.0, 1: 1.0}}
    expected = DataFrame(expected_dict, index=np.array([0, 1]), columns=df.columns)
    tm.assert_frame_equal(result, expected)
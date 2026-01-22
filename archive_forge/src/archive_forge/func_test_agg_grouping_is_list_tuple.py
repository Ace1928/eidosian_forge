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
def test_agg_grouping_is_list_tuple(ts):
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=Index(list('ABCD'), dtype=object), index=pd.date_range('2000-01-01', periods=30, freq='B'))
    grouped = df.groupby(lambda x: x.year)
    grouper = grouped._grouper.groupings[0].grouping_vector
    grouped._grouper.groupings[0] = Grouping(ts.index, list(grouper))
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)
    grouped._grouper.groupings[0] = Grouping(ts.index, tuple(grouper))
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)
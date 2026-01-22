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
def test_groupby_aggregate_empty_with_multiindex_frame():
    df = DataFrame(columns=['a', 'b', 'c'])
    result = df.groupby(['a', 'b'], group_keys=False).agg(d=('c', list))
    expected = DataFrame(columns=['d'], index=MultiIndex([[], []], [[], []], names=['a', 'b']))
    tm.assert_frame_equal(result, expected)
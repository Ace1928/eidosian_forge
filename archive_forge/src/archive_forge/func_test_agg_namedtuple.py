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
def test_agg_namedtuple(self):
    df = DataFrame({'A': [0, 1], 'B': [1, 2]})
    result = df.groupby('A').agg(b=pd.NamedAgg('B', 'sum'), c=pd.NamedAgg(column='B', aggfunc='count'))
    expected = df.groupby('A').agg(b=('B', 'sum'), c=('B', 'count'))
    tm.assert_frame_equal(result, expected)
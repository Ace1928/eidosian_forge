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
def test_groupby_agg_dict_with_getitem():
    dat = DataFrame({'A': ['A', 'A', 'B', 'B', 'B'], 'B': [1, 2, 1, 1, 2]})
    result = dat.groupby('A')[['B']].agg({'B': 'sum'})
    expected = DataFrame({'B': [3, 4]}, index=['A', 'B']).rename_axis('A', axis=0)
    tm.assert_frame_equal(result, expected)
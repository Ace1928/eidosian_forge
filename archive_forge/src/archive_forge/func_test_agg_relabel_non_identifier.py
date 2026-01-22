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
def test_agg_relabel_non_identifier(self):
    df = DataFrame({'group': ['a', 'a', 'b', 'b'], 'A': [0, 1, 2, 3], 'B': [5, 6, 7, 8]})
    result = df.groupby('group').agg(**{'my col': ('A', 'max')})
    expected = DataFrame({'my col': [1, 3]}, index=Index(['a', 'b'], name='group'))
    tm.assert_frame_equal(result, expected)
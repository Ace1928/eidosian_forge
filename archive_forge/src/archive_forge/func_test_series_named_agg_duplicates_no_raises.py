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
def test_series_named_agg_duplicates_no_raises(self):
    gr = Series([1, 2, 3]).groupby([0, 0, 1])
    grouped = gr.agg(a='sum', b='sum')
    expected = DataFrame({'a': [3, 3], 'b': [3, 3]}, index=np.array([0, 1]))
    tm.assert_frame_equal(expected, grouped)
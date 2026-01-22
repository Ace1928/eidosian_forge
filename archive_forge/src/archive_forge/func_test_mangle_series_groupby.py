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
def test_mangle_series_groupby(self):
    gr = Series([1, 2, 3, 4]).groupby([0, 0, 1, 1])
    result = gr.agg([lambda x: 0, lambda x: 1])
    exp_data = {'<lambda_0>': [0, 0], '<lambda_1>': [1, 1]}
    expected = DataFrame(exp_data, index=np.array([0, 1]))
    tm.assert_frame_equal(result, expected)
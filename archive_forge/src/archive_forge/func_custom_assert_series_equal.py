import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def custom_assert_series_equal(left, right, *args, **kwargs):
    if left.dtype.name == 'json':
        assert left.dtype == right.dtype
        left = pd.Series(JSONArray(left.values.astype(object)), index=left.index, name=left.name)
        right = pd.Series(JSONArray(right.values.astype(object)), index=right.index, name=right.name)
    tm.assert_series_equal(left, right, *args, **kwargs)
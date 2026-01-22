from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_different_index_names():
    left = DataFrame({'a': [1]}, index=Index([1], name='c'))
    right = DataFrame({'a': [1]}, index=Index([1], name='d'))
    result = merge(left, right, left_on='c', right_on='d')
    expected = DataFrame({'a_x': [1], 'a_y': 1})
    tm.assert_frame_equal(result, expected)
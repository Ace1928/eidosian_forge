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
def test_merge_ea_and_non_ea(any_numeric_ea_dtype, join_type):
    left = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
    right = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype.lower())
    result = left.merge(right, how=join_type)
    expected = DataFrame({'a': Series([1, 2, 3], dtype=any_numeric_ea_dtype), 'b': Series([1, 1, 1], dtype=any_numeric_ea_dtype), 'c': Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower())})
    tm.assert_frame_equal(result, expected)
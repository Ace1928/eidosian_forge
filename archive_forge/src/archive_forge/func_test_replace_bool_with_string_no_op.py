import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_bool_with_string_no_op(self):
    s = pd.Series([True, False, True])
    result = s.replace('fun', 'in-the-sun')
    tm.assert_series_equal(s, result)
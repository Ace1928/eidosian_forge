import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_bool_in_mixed_frame(self):
    df = DataFrame({'string_data': ['a', 'b', 'c', 'd', 'e'], 'bool_data': [True, True, False, False, False], 'int_data': [10, 20, 30, 40, 50]})
    result = df.describe()
    expected = DataFrame({'int_data': [5, 30, df.int_data.std(), 10, 20, 30, 40, 50]}, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_frame_equal(result, expected)
    result = df.describe(include=['bool'])
    expected = DataFrame({'bool_data': [5, 2, False, 3]}, index=['count', 'unique', 'top', 'freq'])
    tm.assert_frame_equal(result, expected)
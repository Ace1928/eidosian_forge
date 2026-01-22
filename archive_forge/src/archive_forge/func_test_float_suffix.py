import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_float_suffix(self):
    df = DataFrame({'treatment_1.1': [1.0, 2.0], 'treatment_2.1': [3.0, 4.0], 'result_1.2': [5.0, 6.0], 'result_1': [0, 9], 'A': ['X1', 'X2']})
    expected = DataFrame({'A': ['X1', 'X2', 'X1', 'X2', 'X1', 'X2', 'X1', 'X2'], 'colname': [1.2, 1.2, 1.0, 1.0, 1.1, 1.1, 2.1, 2.1], 'result': [5.0, 6.0, 0.0, 9.0, np.nan, np.nan, np.nan, np.nan], 'treatment': [np.nan, np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]})
    expected = expected.set_index(['A', 'colname'])
    result = wide_to_long(df, ['result', 'treatment'], i='A', j='colname', suffix='[0-9.]+', sep='_')
    tm.assert_frame_equal(result, expected)
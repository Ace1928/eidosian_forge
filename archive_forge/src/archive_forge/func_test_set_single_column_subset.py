import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_single_column_subset(self):
    df = DataFrame({'A': [1, 2, 3], 'B': list('abc'), 'C': [4, np.nan, 5]})
    expected = DataFrame({'A': [1, 3], 'B': list('ac'), 'C': [4.0, 5.0]}, index=[0, 2])
    result = df.dropna(subset='C')
    tm.assert_frame_equal(result, expected)
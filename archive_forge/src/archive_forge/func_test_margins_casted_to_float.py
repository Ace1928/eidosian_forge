from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_margins_casted_to_float(self):
    df = DataFrame({'A': [2, 4, 6, 8], 'B': [1, 4, 5, 8], 'C': [1, 3, 4, 6], 'D': ['X', 'X', 'Y', 'Y']})
    result = pivot_table(df, index='D', margins=True)
    expected = DataFrame({'A': [3.0, 7.0, 5], 'B': [2.5, 6.5, 4.5], 'C': [2.0, 5.0, 3.5]}, index=Index(['X', 'Y', 'All'], name='D'))
    tm.assert_frame_equal(result, expected)
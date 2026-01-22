import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_tuples(self):
    df = DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
    df['C'] = list(zip(df['A'], df['B']))
    result = df['C'].isin([(1, 'a')])
    tm.assert_series_equal(result, Series([True, False, False], name='C'))
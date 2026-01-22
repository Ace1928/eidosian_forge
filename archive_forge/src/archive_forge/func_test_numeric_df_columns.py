import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('columns', ['a', ['a', 'b']])
def test_numeric_df_columns(columns):
    df = DataFrame({'a': [1.2, decimal.Decimal(3.14), decimal.Decimal('infinity'), '0.1'], 'b': [1.0, 2.0, 3.0, 4.0]})
    expected = DataFrame({'a': [1.2, 3.14, np.inf, 0.1], 'b': [1.0, 2.0, 3.0, 4.0]})
    df_copy = df.copy()
    df_copy[columns] = df_copy[columns].apply(to_numeric)
    tm.assert_frame_equal(df_copy, expected)
import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_on_float_by_int(self):
    df1 = pd.DataFrame({'symbol': list('AAABBBCCC'), 'exch': [1, 2, 3, 1, 2, 3, 1, 2, 3], 'price': [3.26, 3.2599, 3.2598, 12.58, 12.59, 12.5, 378.15, 378.2, 378.25]}, columns=['symbol', 'exch', 'price'])
    df2 = pd.DataFrame({'exch': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'price': [0.0, 1.0, 100.0, 0.0, 5.0, 100.0, 0.0, 5.0, 1000.0], 'mpv': [0.0001, 0.01, 0.05, 0.0001, 0.01, 0.1, 0.0001, 0.25, 1.0]}, columns=['exch', 'price', 'mpv'])
    df1 = df1.sort_values('price').reset_index(drop=True)
    df2 = df2.sort_values('price').reset_index(drop=True)
    result = merge_asof(df1, df2, on='price', by='exch')
    expected = pd.DataFrame({'symbol': list('AAABBBCCC'), 'exch': [3, 2, 1, 3, 1, 2, 1, 2, 3], 'price': [3.2598, 3.2599, 3.26, 12.5, 12.58, 12.59, 378.15, 378.2, 378.25], 'mpv': [0.0001, 0.0001, 0.01, 0.25, 0.01, 0.01, 0.05, 0.1, 0.25]}, columns=['symbol', 'exch', 'price', 'mpv'])
    tm.assert_frame_equal(result, expected)
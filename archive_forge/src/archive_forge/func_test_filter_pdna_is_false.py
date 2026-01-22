from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_pdna_is_false():
    df = DataFrame({'A': np.arange(8), 'B': list('aabbbbcc'), 'C': np.arange(8)})
    ser = df['B']
    g_df = df.groupby(df['B'])
    g_s = ser.groupby(ser)
    func = lambda x: pd.NA
    res = g_df.filter(func)
    tm.assert_frame_equal(res, df.loc[[]])
    res = g_s.filter(func)
    tm.assert_series_equal(res, ser[[]])
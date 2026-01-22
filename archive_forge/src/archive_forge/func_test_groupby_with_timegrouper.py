from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_with_timegrouper(self):
    df_original = DataFrame({'Buyer': 'Carl Carl Carl Carl Joe Carl'.split(), 'Quantity': [18, 3, 5, 1, 9, 3], 'Date': [datetime(2013, 9, 1, 13, 0), datetime(2013, 9, 1, 13, 5), datetime(2013, 10, 1, 20, 0), datetime(2013, 10, 3, 10, 0), datetime(2013, 12, 2, 12, 0), datetime(2013, 9, 2, 14, 0)]})
    df_reordered = df_original.sort_values(by='Quantity')
    for df in [df_original, df_reordered]:
        df = df.set_index(['Date'])
        exp_dti = date_range('20130901', '20131205', freq='5D', name='Date', inclusive='left', unit=df.index.unit)
        expected = DataFrame({'Buyer': 0, 'Quantity': 0}, index=exp_dti)
        expected = expected.astype({'Buyer': object})
        expected.iloc[0, 0] = 'CarlCarlCarl'
        expected.iloc[6, 0] = 'CarlCarl'
        expected.iloc[18, 0] = 'Joe'
        expected.iloc[[0, 6, 18], 1] = np.array([24, 6, 9], dtype='int64')
        result1 = df.resample('5D').sum()
        tm.assert_frame_equal(result1, expected)
        df_sorted = df.sort_index()
        result2 = df_sorted.groupby(Grouper(freq='5D')).sum()
        tm.assert_frame_equal(result2, expected)
        result3 = df.groupby(Grouper(freq='5D')).sum()
        tm.assert_frame_equal(result3, expected)
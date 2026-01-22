import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_datetimes(self):
    df = DataFrame(['a', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'g'], columns=['A'], index=date_range('20130101', periods=9))
    dts = [Timestamp(x) for x in ['2004-02-11', '2004-01-21', '2004-01-26', '2005-09-20', '2010-10-04', '2009-05-12', '2008-11-12', '2010-09-28', '2010-09-28']]
    df['B'] = dts[::2] + dts[1::2]
    df['C'] = 2.0
    df['A1'] = 3.0
    df1 = df.sort_values(by='A')
    df2 = df.sort_values(by=['A'])
    tm.assert_frame_equal(df1, df2)
    df1 = df.sort_values(by='B')
    df2 = df.sort_values(by=['B'])
    tm.assert_frame_equal(df1, df2)
    df1 = df.sort_values(by='B')
    df2 = df.sort_values(by=['C', 'B'])
    tm.assert_frame_equal(df1, df2)
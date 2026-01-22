import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_exclude_using_scalars(self):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.Categorical(list('abc')), 'g': pd.date_range('20130101', periods=3), 'h': pd.date_range('20130101', periods=3, tz='US/Eastern'), 'i': pd.date_range('20130101', periods=3, tz='CET'), 'j': pd.period_range('2013-01', periods=3, freq='M'), 'k': pd.timedelta_range('1 day', periods=3)})
    ri = df.select_dtypes(exclude=np.number)
    ei = df[['a', 'e', 'f', 'g', 'h', 'i', 'j']]
    tm.assert_frame_equal(ri, ei)
    ri = df.select_dtypes(exclude='category')
    ei = df[['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k']]
    tm.assert_frame_equal(ri, ei)
    with pytest.raises(NotImplementedError, match='^$'):
        df.select_dtypes(exclude='period')
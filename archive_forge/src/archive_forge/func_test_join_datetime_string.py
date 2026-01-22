import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_datetime_string(self):
    dfa = DataFrame([['2012-08-02', 'L', 10], ['2012-08-02', 'J', 15], ['2013-04-06', 'L', 20], ['2013-04-06', 'J', 25]], columns=['x', 'y', 'a'])
    dfa['x'] = pd.to_datetime(dfa['x']).astype('M8[ns]')
    dfb = DataFrame([['2012-08-02', 'J', 1], ['2013-04-06', 'L', 2]], columns=['x', 'y', 'z'], index=[2, 4])
    dfb['x'] = pd.to_datetime(dfb['x']).astype('M8[ns]')
    result = dfb.join(dfa.set_index(['x', 'y']), on=['x', 'y'])
    expected = DataFrame([[Timestamp('2012-08-02 00:00:00'), 'J', 1, 15], [Timestamp('2013-04-06 00:00:00'), 'L', 2, 20]], index=[2, 4], columns=['x', 'y', 'z', 'a'])
    expected['x'] = expected['x'].astype('M8[ns]')
    tm.assert_frame_equal(result, expected)
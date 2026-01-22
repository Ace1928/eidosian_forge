import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_nested_EA():
    s = pd.Series([pd.date_range('20170101', periods=3, tz='UTC'), pd.date_range('20170104', periods=3, tz='UTC')])
    result = s.explode()
    expected = pd.Series(pd.date_range('20170101', periods=6, tz='UTC'), index=[0, 0, 0, 1, 1, 1])
    tm.assert_series_equal(result, expected)
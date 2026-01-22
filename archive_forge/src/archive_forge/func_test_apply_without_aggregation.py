from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('func', [lambda x: x.resample('20min', group_keys=False), lambda x: x.groupby(pd.Grouper(freq='20min'), group_keys=False)], ids=['resample', 'groupby'])
def test_apply_without_aggregation(func, _test_series):
    t = func(_test_series)
    result = t.apply(lambda x: x)
    tm.assert_series_equal(result, _test_series)
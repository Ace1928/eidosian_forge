from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_partial(self):
    rng = period_range('2007-01', periods=50, freq='M')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    with pytest.raises(KeyError, match="^'2006'$"):
        ts['2006']
    result = ts['2008']
    assert (result.index.year == 2008).all()
    result = ts['2008':'2009']
    assert len(result) == 24
    result = ts['2008-1':'2009-12']
    assert len(result) == 24
    result = ts['2008Q1':'2009Q4']
    assert len(result) == 24
    result = ts[:'2009']
    assert len(result) == 36
    result = ts['2009':]
    assert len(result) == 50 - 24
    exp = result
    result = ts[24:]
    tm.assert_series_equal(exp, result)
    ts = pd.concat([ts[10:], ts[10:]])
    msg = "left slice bound for non-unique label: '2008'"
    with pytest.raises(KeyError, match=msg):
        ts[slice('2008', '2009')]
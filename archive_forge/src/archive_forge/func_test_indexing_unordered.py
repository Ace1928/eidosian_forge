from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_indexing_unordered():
    rng = date_range(start='2011-01-01', end='2011-01-15')
    ts = Series(np.random.default_rng(2).random(len(rng)), index=rng)
    ts2 = pd.concat([ts[0:4], ts[-4:], ts[4:-4]])
    for t in ts.index:
        expected = ts[t]
        result = ts2[t]
        assert expected == result

    def compare(slobj):
        result = ts2[slobj].copy()
        result = result.sort_index()
        expected = ts[slobj]
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)
    for key in [slice('2011-01-01', '2011-01-15'), slice('2010-12-30', '2011-01-15'), slice('2011-01-01', '2011-01-16'), slice('2011-01-01', '2011-01-6'), slice('2011-01-06', '2011-01-8'), slice('2011-01-06', '2011-01-12')]:
        with pytest.raises(KeyError, match='Value based partial slicing on non-monotonic'):
            compare(key)
    result = ts2['2011'].sort_index()
    expected = ts['2011']
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)
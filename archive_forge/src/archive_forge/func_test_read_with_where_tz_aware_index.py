from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_read_with_where_tz_aware_index(tmp_path, setup_path):
    periods = 10
    dts = date_range('20151201', periods=periods, freq='D', tz='UTC')
    mi = pd.MultiIndex.from_arrays([dts, range(periods)], names=['DATE', 'NO'])
    expected = DataFrame({'MYCOL': 0}, index=mi)
    key = 'mykey'
    path = tmp_path / setup_path
    with pd.HDFStore(path) as store:
        store.append(key, expected, format='table', append=True)
    result = pd.read_hdf(path, key, where='DATE > 20151130')
    tm.assert_frame_equal(result, expected)
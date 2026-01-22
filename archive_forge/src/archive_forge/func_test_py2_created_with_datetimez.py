from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_py2_created_with_datetimez(datapath):
    index = DatetimeIndex(['2019-01-01T18:00'], dtype='M8[ns, America/New_York]')
    expected = DataFrame({'data': 123}, index=index)
    with ensure_clean_store(datapath('io', 'data', 'legacy_hdf', 'gh26443.h5'), mode='r') as store:
        result = store['key']
        tm.assert_frame_equal(result, expected)
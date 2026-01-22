from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
@td.skip_if_windows
def test_store_timezone(setup_path):
    with ensure_clean_store(setup_path) as store:
        today = date(2013, 9, 10)
        df = DataFrame([1, 2, 3], index=[today, today, today])
        store['obj1'] = df
        result = store['obj1']
        tm.assert_frame_equal(result, df)
    with ensure_clean_store(setup_path) as store:
        with tm.set_timezone('EST5EDT'):
            today = date(2013, 9, 10)
            df = DataFrame([1, 2, 3], index=[today, today, today])
            store['obj1'] = df
        with tm.set_timezone('CST6CDT'):
            result = store['obj1']
        tm.assert_frame_equal(result, df)
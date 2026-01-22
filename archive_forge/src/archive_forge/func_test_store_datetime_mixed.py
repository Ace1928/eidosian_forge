import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_store_datetime_mixed(setup_path):
    df = DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': ['a', 'b', 'c']})
    ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    df['d'] = ts.index[:3]
    _check_roundtrip(df, tm.assert_frame_equal, path=setup_path)
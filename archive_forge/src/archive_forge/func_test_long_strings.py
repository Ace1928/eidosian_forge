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
def test_long_strings(setup_path):
    data = ['a' * 50] * 10
    df = DataFrame({'a': data}, index=data)
    with ensure_clean_store(setup_path) as store:
        store.append('df', df, data_columns=['a'])
        result = store.select('df')
        tm.assert_frame_equal(df, result)
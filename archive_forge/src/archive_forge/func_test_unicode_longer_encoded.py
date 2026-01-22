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
def test_unicode_longer_encoded(setup_path):
    char = 'Δ'
    df = DataFrame({'A': [char]})
    with ensure_clean_store(setup_path) as store:
        store.put('df', df, format='table', encoding='utf-8')
        result = store.get('df')
        tm.assert_frame_equal(result, df)
    df = DataFrame({'A': ['a', char], 'B': ['b', 'b']})
    with ensure_clean_store(setup_path) as store:
        store.put('df', df, format='table', encoding='utf-8')
        result = store.get('df')
        tm.assert_frame_equal(result, df)
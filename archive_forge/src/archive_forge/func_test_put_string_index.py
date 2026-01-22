import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_put_string_index(setup_path):
    with ensure_clean_store(setup_path) as store:
        index = Index([f'I am a very long string index: {i}' for i in range(20)])
        s = Series(np.arange(20), index=index)
        df = DataFrame({'A': s, 'B': s})
        store['a'] = s
        tm.assert_series_equal(store['a'], s)
        store['b'] = df
        tm.assert_frame_equal(store['b'], df)
        index = Index(['abcdefghijklmnopqrstuvwxyz1234567890'] + [f'I am a very long string index: {i}' for i in range(20)])
        s = Series(np.arange(21), index=index)
        df = DataFrame({'A': s, 'B': s})
        store['a'] = s
        tm.assert_series_equal(store['a'], s)
        store['b'] = df
        tm.assert_frame_equal(store['b'], df)
import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_mi_data_columns(setup_path):
    idx = MultiIndex.from_arrays([date_range('2000-01-01', periods=5), range(5)], names=['date', 'id'])
    df = DataFrame({'a': [1.1, 1.2, 1.3, 1.4, 1.5]}, index=idx)
    with ensure_clean_store(setup_path) as store:
        store.append('df', df, data_columns=True)
        actual = store.select('df', where='id == 1')
        expected = df.iloc[[1], :]
        tm.assert_frame_equal(actual, expected)
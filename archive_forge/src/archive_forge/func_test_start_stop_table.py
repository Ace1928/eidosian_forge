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
def test_start_stop_table(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'A': np.random.default_rng(2).random(20), 'B': np.random.default_rng(2).random(20)})
        store.append('df', df)
        result = store.select('df', "columns=['A']", start=0, stop=5)
        expected = df.loc[0:4, ['A']]
        tm.assert_frame_equal(result, expected)
        result = store.select('df', "columns=['A']", start=30, stop=40)
        assert len(result) == 0
        expected = df.loc[30:40, ['A']]
        tm.assert_frame_equal(result, expected)
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
def test_start_stop_multiple(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'foo': [1, 2], 'bar': [1, 2]})
        store.append_to_multiple({'selector': ['foo'], 'data': None}, df, selector='selector')
        result = store.select_as_multiple(['selector', 'data'], selector='selector', start=0, stop=1)
        expected = df.loc[[0], ['foo', 'bar']]
        tm.assert_frame_equal(result, expected)
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
def test_store_bool_index(tmp_path, setup_path):
    df = DataFrame([[1]], columns=[True], index=Index([False], dtype='bool'))
    expected = df.copy()
    path = tmp_path / setup_path
    df.to_hdf(path, key='a')
    result = read_hdf(path, 'a')
    tm.assert_frame_equal(expected, result)
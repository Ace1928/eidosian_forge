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
def test_duplicate_column_name(tmp_path, setup_path):
    df = DataFrame(columns=['a', 'a'], data=[[0, 0]])
    path = tmp_path / setup_path
    msg = 'Columns index has to be unique for fixed format'
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key='df', format='fixed')
    df.to_hdf(path, key='df', format='table')
    other = read_hdf(path, 'df')
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)
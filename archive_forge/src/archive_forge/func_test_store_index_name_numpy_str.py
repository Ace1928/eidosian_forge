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
@pytest.mark.parametrize('tz', [None, 'US/Pacific'])
@pytest.mark.parametrize('table_format', ['table', 'fixed'])
def test_store_index_name_numpy_str(tmp_path, table_format, setup_path, unit, tz):
    idx = DatetimeIndex([dt.date(2000, 1, 1), dt.date(2000, 1, 2)], name='colsג').tz_localize(tz)
    idx1 = DatetimeIndex([dt.date(2010, 1, 1), dt.date(2010, 1, 2)], name='rowsא').as_unit(unit).tz_localize(tz)
    df = DataFrame(np.arange(4).reshape(2, 2), columns=idx, index=idx1)
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', format=table_format)
    df2 = read_hdf(path, 'df')
    tm.assert_frame_equal(df, df2, check_names=True)
    assert isinstance(df2.index.name, str)
    assert isinstance(df2.columns.name, str)
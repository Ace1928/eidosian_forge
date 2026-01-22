from contextlib import closing
from pathlib import Path
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
from pandas.io.pytables import TableIterator
def test_read_infer_string(tmp_path, setup_path):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': ['a', 'b', None]})
    path = tmp_path / setup_path
    df.to_hdf(path, key='data', format='table')
    with pd.option_context('future.infer_string', True):
        result = read_hdf(path, key='data', mode='r')
    expected = DataFrame({'a': ['a', 'b', None]}, dtype='string[pyarrow_numpy]', columns=Index(['a'], dtype='string[pyarrow_numpy]'))
    tm.assert_frame_equal(result, expected)